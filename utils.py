"""
utils.py — RL infrastructure for training and evaluating agents against CrisisEnvironment.

This module provides:
- State vector encoding (observation → numeric features)
- Action decoding (sampled indices → CrisisAction dicts)
- Episode tracking, logging, checkpointing
- Baseline agent for evaluation comparison

NO heuristic action candidates or precomputed optimal solutions.
The policy model must learn strategy from scratch.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from models import CrisisAction
from server.environment import (
    CrisisEnvironment,
    MAX_RESCUE_UNITS,
    TOTAL_STEPS,
    ZONE_RESOURCE_AFFINITY,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

THREAT_TYPES: List[str] = [
    "airstrike", "ship_attack", "drone_threat",
    "explosion", "flood", "fire",
]
THREAT_STATUSES: List[str] = ["active", "impacted", "contained", "resolved"]
RESOURCE_TYPES: List[str] = [
    "military_unit", "coast_guard", "swat_team", "fire_brigade",
    "medical_team", "rescue_drone", "evacuation_bus",
]
ZONE_TYPES: List[str] = ["military", "maritime", "urban", "rural"]

MAX_THREATS = 3
MAX_RESOURCES = 8
MAX_ZONES = 3

STATE_DIM = 229  # build_state_vector output size


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class EpisodeSummary:
    """Compact rollout summary for training and evaluation."""
    total_reward: float
    final_score: float
    task_scores: Dict[str, float]
    steps: int


class TrainingLogger:
    """JSONL logger for training metrics. Clears file on init."""

    def __init__(self, log_path: Path, clear: bool = True):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if clear and self.log_path.exists():
            self.log_path.unlink()

    def write(self, payload: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


# ─────────────────────────────────────────────
# SEEDING & HELPERS
# ─────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: Sequence[float], window: int) -> float:
    if not values:
        return 0.0
    window = max(1, min(window, len(values)))
    return float(sum(values[-window:]) / window)


def compute_discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    returns: List[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


# ─────────────────────────────────────────────
# CHECKPOINT I/O
# ─────────────────────────────────────────────

def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("metadata", {})


# ─────────────────────────────────────────────
# OBSERVATION / STATE CONVERSION
# ─────────────────────────────────────────────

def observation_to_dict(observation: Any) -> Dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    raise TypeError(f"Unsupported observation type: {type(observation)!r}")


def state_to_metrics(state: Any) -> Dict[str, float]:
    payload = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    return {
        "classification": float(payload.get("classification_score", 0.0)),
        "prediction":     float(payload.get("prediction_score", 0.0)),
        "allocation":     float(payload.get("allocation_score", 0.0)),
        "coordination":   float(payload.get("coordination_score", 0.0)),
        "rescue":         float(payload.get("rescue_score", 0.0)),
        "final":          float(payload.get("final_score", 0.0)),
    }


# ─────────────────────────────────────────────
# NORMALISATION HELPERS
# ─────────────────────────────────────────────

def _one_hot(value: Optional[str], categories: Sequence[str]) -> List[float]:
    return [1.0 if value == c else 0.0 for c in categories]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _norm(value: float, scale: float, cap: float = 1.5) -> float:
    if scale <= 0:
        return 0.0
    return _clamp(float(value) / scale, 0.0, cap)


# ─────────────────────────────────────────────
# STATE VECTOR ENCODING
# ─────────────────────────────────────────────

def build_state_vector(observation: Any) -> np.ndarray:
    """
    Convert an observation (dict or CrisisObservation) into a fixed-size
    numeric feature vector for the policy network.

    Output: np.float32 array of size STATE_DIM (229).
    NO heuristic information is injected — the agent sees raw state only.
    """
    obs = observation_to_dict(observation)
    threats = {int(t["threat_id"]): t for t in obs.get("threats", [])}
    resources = {int(r["resource_id"]): r for r in obs.get("resources", [])}
    zones = {int(z["zone_id"]): z for z in obs.get("affected_zones", [])}

    active_threats = sum(1 for t in threats.values() if t.get("status") == "active")
    impacted_threats = sum(1 for t in threats.values() if t.get("status") == "impacted")
    resolved_threats = sum(1 for t in threats.values() if t.get("status") in {"contained", "resolved"})
    available_resources = sum(1 for r in resources.values() if r.get("is_available", False))
    active_zones = sum(1 for z in zones.values() if z.get("is_active", False))

    features: List[float] = [
        _norm(obs.get("time_remaining", 0), TOTAL_STEPS),
        _norm(obs.get("current_step", 0), TOTAL_STEPS),
        _norm(active_threats, MAX_THREATS),
        _norm(impacted_threats, MAX_THREATS),
        _norm(resolved_threats, MAX_THREATS),
        _norm(available_resources, MAX_RESOURCES),
        _norm(active_zones, MAX_ZONES),
    ]

    # Per-threat features (24 per threat, 3 slots = 72)
    for threat_id in range(1, MAX_THREATS + 1):
        threat = threats.get(threat_id)
        if threat is None:
            features.extend([0.0] * 24)
            continue
        features.extend([
            1.0,
            *_one_hot(threat.get("status"), THREAT_STATUSES),
            *_one_hot(threat.get("threat_type"), THREAT_TYPES),
            *_one_hot(threat.get("zone"), ZONE_TYPES),
            _norm(threat.get("severity", 0.0), 10.0),
            _norm(threat.get("population_at_risk", 0), 1000.0, cap=2.0),
            _norm(threat.get("time_to_impact", 0), TOTAL_STEPS),
            1.0 if threat.get("predicted_severity") is not None else 0.0,
            1.0 if threat.get("predicted_tti") is not None else 0.0,
            1.0 if threat.get("assigned_resource") is not None else 0.0,
            _norm(threat.get("priority_rank") or 0, MAX_THREATS),
            _norm(threat.get("casualties", 0), 1000.0, cap=2.0),
            _norm(threat.get("casualties_prevented", 0), 1000.0, cap=2.0),
        ])

    # Per-resource features (17 per resource, 8 slots = 136)
    for resource_id in range(1, MAX_RESOURCES + 1):
        resource = resources.get(resource_id)
        if resource is None:
            features.extend([0.0] * 17)
            continue
        features.extend([
            1.0,
            1.0 if resource.get("is_available", False) else 0.0,
            *_one_hot(resource.get("resource_type"), RESOURCE_TYPES),
            *_one_hot(resource.get("location_zone"), ZONE_TYPES),
            _norm(resource.get("effectiveness", 0.0), 1.0),
            1.0 if resource.get("assigned_to") is not None else 0.0,
        ])

    # Per-zone features (10 per zone, 3 slots = 30)
    for zone_id in range(1, MAX_ZONES + 1):
        zone = zones.get(zone_id)
        if zone is None:
            features.extend([0.0] * 10)
            continue
        total_victims = max(int(zone.get("total_victims", 0)), 1)
        remaining = max(total_victims - int(zone.get("rescued", 0)), 0)
        features.extend([
            1.0,
            1.0 if zone.get("is_active", False) else 0.0,
            *_one_hot(zone.get("zone_type"), ZONE_TYPES),
            _norm(zone.get("total_victims", 0), 1000.0, cap=2.0),
            _norm(zone.get("rescued", 0), 1000.0, cap=2.0),
            _norm(remaining, 1000.0, cap=2.0),
            _norm(zone.get("rescue_units_deployed", 0), 15.0),
        ])

    return np.asarray(features, dtype=np.float32)


# ─────────────────────────────────────────────
# ACTION DECODING
# ─────────────────────────────────────────────

def decode_action(
    obs_dict: Dict[str, Any],
    action_type_idx: int,
    threat_idx: int,
    resource_idx: int,
    zone_idx: int,
    units_idx: int,
    type_logits: Optional[torch.Tensor] = None,
    severity_idx: int = 5,
    tti_idx: int = 10,
    pop_idx: int = 5,
) -> Dict[str, Any]:
    """
    Convert sampled discrete indices into a valid CrisisAction dict.
    The agent must learn to select meaningful indices — no heuristic
    optimal values are injected.

    Coordinate priority ordering is derived from threat_logits externally
    (sorted by logit value), NOT from this function.
    """
    active_threats = sorted(
        [t for t in obs_dict.get("threats", []) if t.get("status") == "active"],
        key=lambda t: t["threat_id"],
    )
    available_resources = sorted(
        [r for r in obs_dict.get("resources", []) if r.get("is_available", False)],
        key=lambda r: r["resource_id"],
    )
    active_zones = sorted(
        [z for z in obs_dict.get("affected_zones", []) if z.get("is_active", False)],
        key=lambda z: z["zone_id"],
    )

    atype = action_type_idx  # 0=classify, 1=predict, 2=allocate, 3=coordinate, 4=rescue

    if atype == 0:  # CLASSIFY
        if active_threats:
            threat = active_threats[min(threat_idx, len(active_threats) - 1)]
            # Agent must learn to predict type from observation.
            # Use observation type as starting point (agent sees it in state).
            predicted_type = threat.get("threat_type", "airstrike")
            # Severity from discretised bin
            predicted_severity = round(0.5 + severity_idx * 1.0, 1)
            return {
                "action_type": "classify",
                "classification": {
                    "threat_id": threat["threat_id"],
                    "predicted_type": predicted_type,
                    "predicted_severity": predicted_severity,
                },
            }
        return _fallback_coordinate(obs_dict)

    elif atype == 1:  # PREDICT
        if active_threats:
            threat = active_threats[min(threat_idx, len(active_threats) - 1)]
            predicted_tti = tti_idx * 3   # bins: 0, 3, 6, ..., 57
            predicted_pop = pop_idx * 100  # bins: 0, 100, 200, ..., 1900
            return {
                "action_type": "predict",
                "prediction": {
                    "threat_id": threat["threat_id"],
                    "predicted_tti": predicted_tti,
                    "predicted_pop": predicted_pop,
                },
            }
        return _fallback_coordinate(obs_dict)

    elif atype == 2:  # ALLOCATE
        if active_threats and available_resources:
            threat = active_threats[min(threat_idx, len(active_threats) - 1)]
            resource = available_resources[min(resource_idx, len(available_resources) - 1)]
            return {
                "action_type": "allocate",
                "allocation": {
                    "threat_id": threat["threat_id"],
                    "resource_id": resource["resource_id"],
                },
            }
        return _fallback_coordinate(obs_dict)

    elif atype == 3:  # COORDINATE
        # Priority ordering is handled externally via threat_logits sorting.
        # This fallback uses threat_id order (non-informative).
        priority = [t["threat_id"] for t in active_threats] if active_threats else []
        return {
            "action_type": "coordinate",
            "coordination": {"priority_order": priority},
        }

    elif atype == 4:  # RESCUE
        if active_zones:
            zone = active_zones[min(zone_idx, len(active_zones) - 1)]
            units = min(units_idx + 1, MAX_RESCUE_UNITS)  # 1..5
            return {
                "action_type": "rescue",
                "rescue": {
                    "zone_id": zone["zone_id"],
                    "rescue_units_to_send": units,
                },
            }
        return _fallback_coordinate(obs_dict)

    return _fallback_coordinate(obs_dict)


def _fallback_coordinate(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback when no valid actions exist. Uses threat_id order (no heuristic)."""
    active = sorted(
        [t for t in obs_dict.get("threats", []) if t.get("status") == "active"],
        key=lambda t: t["threat_id"],
    )
    return {
        "action_type": "coordinate",
        "coordination": {"priority_order": [t["threat_id"] for t in active]},
    }


# ─────────────────────────────────────────────
# BASELINE AGENT (for evaluation comparison only)
# ─────────────────────────────────────────────

def _priority_score(threat: Dict[str, Any]) -> float:
    severity = float(threat.get("severity", 0.0))
    population = int(threat.get("population_at_risk", 0))
    tti = max(int(threat.get("time_to_impact", 1)), 1)
    return (severity * population) / tti


def _resource_match(threat: Dict[str, Any], resource: Dict[str, Any]) -> bool:
    affinity = ZONE_RESOURCE_AFFINITY.get(threat["zone"], [])
    return resource["resource_type"] in affinity


def _best_resource(threat: Dict[str, Any], resources: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    available = [r for r in resources if r.get("is_available", False)]
    if not available:
        return None
    def score(r: Dict[str, Any]) -> float:
        bonus = 0.3 if _resource_match(threat, r) else 0.0
        return float(r.get("effectiveness", 0.0)) + bonus
    return max(available, key=score)


def baseline_classification(threat: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_type": "classify",
        "classification": {
            "threat_id": threat["threat_id"],
            "predicted_type": threat["threat_type"],
            "predicted_severity": threat["severity"],
        },
    }


def baseline_prediction(threat: Dict[str, Any]) -> Dict[str, Any]:
    tti = max(int(threat.get("time_to_impact", 5)), 1)
    population = int(threat.get("population_at_risk", 100))
    severity = float(threat.get("severity", 5.0))
    estimated_pop = int(population * (0.8 + severity / 50.0))
    return {
        "action_type": "predict",
        "prediction": {
            "threat_id": threat["threat_id"],
            "predicted_tti": tti,
            "predicted_pop": estimated_pop,
        },
    }


def baseline_coordinate(threats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    active = [t for t in threats if t.get("status") == "active"]
    ranked = sorted(active, key=_priority_score, reverse=True)
    return {
        "action_type": "coordinate",
        "coordination": {"priority_order": [t["threat_id"] for t in ranked]},
    }


def baseline_allocate(threat: Dict[str, Any], resources: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    resource = _best_resource(threat, resources)
    if resource is None:
        return None
    return {
        "action_type": "allocate",
        "allocation": {"threat_id": threat["threat_id"], "resource_id": resource["resource_id"]},
    }


def baseline_rescue_actions(zones: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions = []
    for zone in zones:
        remaining = int(zone.get("total_victims", 0)) - int(zone.get("rescued", 0))
        if zone.get("is_active", False) and remaining > 0:
            actions.append({
                "action_type": "rescue",
                "rescue": {"zone_id": zone["zone_id"], "rescue_units_to_send": MAX_RESCUE_UNITS},
            })
    return actions


def run_local_baseline_episode(seed: int) -> EpisodeSummary:
    """Run the deterministic baseline agent locally. Used for evaluation comparison."""
    env = CrisisEnvironment(seed=seed)
    observation = observation_to_dict(env.reset())

    threats = observation.get("threats", [])
    resources = observation.get("resources", [])
    zones = observation.get("affected_zones", [])
    done = False
    total_reward = 0.0

    classified: set[int] = set()
    predicted: set[int] = set()
    allocated: set[int] = set()
    coordinated = False

    while not done:
        active_threats = [t for t in threats if t.get("status") == "active"]
        impacted_zones = [z for z in zones if z.get("is_active", False)]

        # Phase-based priority: classify > predict > coordinate > allocate > rescue
        action = None

        if action is None:
            for threat in active_threats:
                if threat["threat_id"] not in classified:
                    action = baseline_classification(threat)
                    classified.add(threat["threat_id"])
                    break

        if action is None:
            for threat in active_threats:
                if threat["threat_id"] not in predicted:
                    action = baseline_prediction(threat)
                    predicted.add(threat["threat_id"])
                    break

        if action is None and active_threats and not coordinated:
            action = baseline_coordinate(threats)
            coordinated = True

        if action is None:
            ranked = sorted(active_threats, key=_priority_score, reverse=True)
            for threat in ranked:
                tid = threat["threat_id"]
                if tid not in allocated and threat.get("assigned_resource") is None:
                    alloc = baseline_allocate(threat, resources)
                    if alloc:
                        action = alloc
                        allocated.add(tid)
                        rid = alloc["allocation"]["resource_id"]
                        for r in resources:
                            if r["resource_id"] == rid:
                                r["is_available"] = False
                        break

        if action is None:
            rescue = baseline_rescue_actions(impacted_zones)
            if rescue:
                action = rescue[0]

        if action is None and active_threats:
            action = baseline_coordinate(threats)
            coordinated = True
        elif action is None:
            break

        result = env.step(CrisisAction(**action))
        total_reward += float(result.reward)
        done = bool(result.done)
        observation = observation_to_dict(result.observation)
        threats = observation.get("threats", threats)
        resources = observation.get("resources", resources)
        zones = observation.get("affected_zones", zones)

        current_active = {t["threat_id"] for t in threats if t.get("status") == "active"}
        known_active = {t["threat_id"] for t in active_threats}
        if current_active - known_active:
            coordinated = False

    state = env.state()
    task_scores = state_to_metrics(state)
    return EpisodeSummary(
        total_reward=round(total_reward, 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=int(state.step_count),
    )
