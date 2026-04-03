"""
utils.py — RL infrastructure for training and evaluating agents against CrisisEnvironment.

Provides:
- Observation/state encoding for policy input
- Action masking helpers
- Action decoding/encoding
- Baseline policy (intentionally imperfect) for comparison and BC warm-start
- Logging/checkpoint helpers
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from models import CrisisAction
from server.environment import (
    ACTION_TYPE_ORDER,
    MAX_RESCUE_UNITS,
    MAX_THREATS_VISIBLE,
    STRATEGIES,
    TOTAL_STEPS,
    ZONE_RESOURCE_AFFINITY,
    CrisisEnvironment,
)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

THREAT_TYPES: List[str] = [
    "airstrike",
    "ship_attack",
    "drone_threat",
    "explosion",
    "flood",
    "fire",
]
THREAT_STATUSES: List[str] = ["active", "impacted", "contained", "resolved"]
RISK_LEVELS: List[str] = ["low", "medium", "high"]
RESOURCE_TYPES: List[str] = [
    "military_unit",
    "coast_guard",
    "swat_team",
    "fire_brigade",
    "medical_team",
    "rescue_drone",
    "evacuation_bus",
]
ZONE_TYPES: List[str] = ["military", "maritime", "urban", "rural"]

ACTION_TYPES: List[str] = list(ACTION_TYPE_ORDER)
STRATEGY_TYPES: List[str] = list(STRATEGIES)

MAX_THREATS = MAX_THREATS_VISIBLE
MAX_RESOURCES = 8
MAX_ZONES = MAX_THREATS_VISIBLE

THREAT_FEATURES = 29
RESOURCE_FEATURES = 15
ZONE_FEATURES = 11
GLOBAL_FEATURES = 14
RECENT_ACTION_STEPS = 4

STATE_DIM = (
    GLOBAL_FEATURES
    + MAX_THREATS * THREAT_FEATURES
    + MAX_RESOURCES * RESOURCE_FEATURES
    + MAX_ZONES * ZONE_FEATURES
    + RECENT_ACTION_STEPS * len(ACTION_TYPES)
    + len(ACTION_TYPES)
    + MAX_RESCUE_UNITS
    + len(STRATEGY_TYPES)
)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────


@dataclass
class EpisodeSummary:
    total_reward: float
    final_score: float
    task_scores: Dict[str, float]
    steps: int


class TrainingLogger:
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
        "prediction": float(payload.get("prediction_score", 0.0)),
        "allocation": float(payload.get("allocation_score", 0.0)),
        "coordination": float(payload.get("coordination_score", 0.0)),
        "rescue": float(payload.get("rescue_score", 0.0)),
        "final": float(payload.get("final_score", 0.0)),
    }


# ─────────────────────────────────────────────
# NUMERIC HELPERS
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
# ACTION MASKING
# ─────────────────────────────────────────────


def build_valid_action_mask(observation: Any) -> Dict[str, Any]:
    obs = observation_to_dict(observation)
    provided = obs.get("valid_actions")

    threats = sorted([t for t in obs.get("threats", []) if t.get("status") == "active"], key=lambda t: int(t["threat_id"]))
    resources = sorted([r for r in obs.get("resources", []) if r.get("is_available", False)], key=lambda r: int(r["resource_id"]))
    zones = sorted([z for z in obs.get("affected_zones", []) if z.get("is_active", False)], key=lambda z: int(z["zone_id"]))

    budget_remaining = int(obs.get("resource_budget_remaining", 0))

    if isinstance(provided, dict) and provided.get("action_mask"):
        raw_action_mask = list(provided.get("action_mask", []))
        if len(raw_action_mask) != len(ACTION_TYPES):
            raw_action_mask = [1] * len(ACTION_TYPES)
    else:
        action_enabled = {
            "classify": int(bool(threats)),
            "predict": int(bool(threats)),
            "allocate": int(bool(threats and resources and budget_remaining > 0)),
            "coordinate": int(len(threats) >= 2),
            "rescue": int(bool(zones and budget_remaining > 0)),
            "skip": 1,
            "delay": int(bool(threats)),
        }
        raw_action_mask = [action_enabled[a] for a in ACTION_TYPES]

    threat_ids = [int(t["threat_id"]) for t in threats[:MAX_THREATS]]
    resource_ids = [int(r["resource_id"]) for r in resources[:MAX_RESOURCES]]
    zone_ids = [int(z["zone_id"]) for z in zones[:MAX_ZONES]]

    threat_mask = [1] * len(threat_ids) + [0] * (MAX_THREATS - len(threat_ids))
    resource_mask = [1] * len(resource_ids) + [0] * (MAX_RESOURCES - len(resource_ids))
    zone_mask = [1] * len(zone_ids) + [0] * (MAX_ZONES - len(zone_ids))

    max_units = int(provided.get("max_rescue_units", 0)) if isinstance(provided, dict) else min(MAX_RESCUE_UNITS, max(0, budget_remaining))
    max_units = max(0, min(max_units, MAX_RESCUE_UNITS))
    units_mask = [1 if i < max_units else 0 for i in range(MAX_RESCUE_UNITS)]

    strategy_mask = [1] * len(STRATEGY_TYPES)

    return {
        "action_types": list(ACTION_TYPES),
        "action_mask": raw_action_mask,
        "threat_ids": threat_ids,
        "threat_mask": threat_mask,
        "resource_ids": resource_ids,
        "resource_mask": resource_mask,
        "zone_ids": zone_ids,
        "zone_mask": zone_mask,
        "units_mask": units_mask,
        "strategy_mask": strategy_mask,
    }


# ─────────────────────────────────────────────
# STATE VECTOR ENCODING
# ─────────────────────────────────────────────


def build_state_vector(observation: Any) -> np.ndarray:
    obs = observation_to_dict(observation)
    mask = build_valid_action_mask(obs)

    threats = {int(t["threat_id"]): t for t in obs.get("threats", [])}
    resources = {int(r["resource_id"]): r for r in obs.get("resources", [])}
    zones = {int(z["zone_id"]): z for z in obs.get("affected_zones", [])}

    active_threats = [t for t in threats.values() if t.get("status") == "active"]
    impacted_threats = [t for t in threats.values() if t.get("status") == "impacted"]
    resolved_threats = [t for t in threats.values() if t.get("status") in {"contained", "resolved"}]
    available_resources = [r for r in resources.values() if r.get("is_available", False)]
    active_zones = [z for z in zones.values() if z.get("is_active", False)]

    mean_sev_unc = float(np.mean([t.get("severity_uncertainty", 0.0) for t in active_threats])) if active_threats else 0.0
    mean_pop_unc = float(np.mean([t.get("population_uncertainty", 0.0) for t in active_threats])) if active_threats else 0.0
    mean_tti_unc = float(np.mean([t.get("tti_uncertainty", 0.0) for t in active_threats])) if active_threats else 0.0

    budget_remaining = float(obs.get("resource_budget_remaining", 0))
    budget_total = float(obs.get("resource_budget_total", 1))

    global_features: List[float] = [
        _norm(obs.get("time_remaining", 0), TOTAL_STEPS),
        _norm(obs.get("current_step", 0), TOTAL_STEPS),
        _norm(len(active_threats), MAX_THREATS),
        _norm(len(impacted_threats), MAX_THREATS),
        _norm(len(resolved_threats), MAX_THREATS),
        _norm(len(available_resources), MAX_RESOURCES),
        _norm(len(active_zones), MAX_ZONES),
        _norm(budget_remaining, max(budget_total, 1.0)),
        _norm(budget_total, 8.0),
        _clamp(mean_sev_unc),
        _clamp(mean_pop_unc),
        _clamp(mean_tti_unc),
        float(mask["action_mask"][ACTION_TYPES.index("rescue")]),
        float(mask["action_mask"][ACTION_TYPES.index("allocate")]),
    ]

    features: List[float] = list(global_features)

    # Threat slots (stable sorted by id)
    for tid in range(1, MAX_THREATS + 1):
        t = threats.get(tid)
        if t is None:
            features.extend([0.0] * THREAT_FEATURES)
            continue

        features.extend([
            1.0,
            *_one_hot(t.get("status"), THREAT_STATUSES),
            *_one_hot(t.get("threat_type"), THREAT_TYPES),
            *_one_hot(t.get("zone"), ZONE_TYPES),
            _norm(t.get("severity", 0.0), 10.0),
            _norm(t.get("population_at_risk", 0), 2000.0, cap=2.0),
            _norm(t.get("time_to_impact", 0), TOTAL_STEPS),
            _clamp(float(t.get("severity_uncertainty", 0.0))),
            _clamp(float(t.get("population_uncertainty", 0.0))),
            _clamp(float(t.get("tti_uncertainty", 0.0))),
            1.0 if t.get("assigned_resource") is not None else 0.0,
            1.0 if t.get("predicted_severity") is not None else 0.0,
            1.0 if t.get("predicted_tti") is not None else 0.0,
            1.0 if t.get("predicted_pop") is not None else 0.0,
            _norm(t.get("priority_score", 0.0), 1200.0, cap=2.0),
            *_one_hot(t.get("risk_level"), RISK_LEVELS),
        ])

    # Resource slots
    for rid in range(1, MAX_RESOURCES + 1):
        r = resources.get(rid)
        if r is None:
            features.extend([0.0] * RESOURCE_FEATURES)
            continue

        features.extend([
            1.0,
            1.0 if r.get("is_available", False) else 0.0,
            *_one_hot(r.get("resource_type"), RESOURCE_TYPES),
            *_one_hot(r.get("location_zone"), ZONE_TYPES),
            _norm(r.get("effectiveness", 0.0), 1.0),
            _norm(r.get("cooldown_steps", 0), 5.0),
        ])

    # Zone slots
    for zid in range(1, MAX_ZONES + 1):
        z = zones.get(zid)
        if z is None:
            features.extend([0.0] * ZONE_FEATURES)
            continue

        total = max(int(z.get("total_victims", 0)), 1)
        rescued = int(z.get("rescued", 0))
        remaining = max(total - rescued, 0)

        features.extend([
            1.0,
            1.0 if z.get("is_active", False) else 0.0,
            *_one_hot(z.get("zone_type"), ZONE_TYPES),
            _norm(total, 1500.0, cap=2.0),
            _norm(rescued, 1500.0, cap=2.0),
            _norm(remaining, 1500.0, cap=2.0),
            _norm(z.get("rescue_units_deployed", 0), 20.0),
            _norm(z.get("evacuated", 0), 1500.0, cap=2.0),
        ])

    # Recent actions memory
    recent = [str(a) for a in obs.get("recent_actions", [])][-RECENT_ACTION_STEPS:]
    recent = ["skip"] * (RECENT_ACTION_STEPS - len(recent)) + recent
    for act in recent:
        features.extend(_one_hot(act, ACTION_TYPES))

    # Action mask, units mask, strategy prior
    features.extend([float(v) for v in mask["action_mask"]])
    features.extend([float(v) for v in mask["units_mask"]])
    features.extend([1.0 / len(STRATEGY_TYPES)] * len(STRATEGY_TYPES))

    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[0] != STATE_DIM:
        raise ValueError(f"State vector size mismatch: expected {STATE_DIM}, got {arr.shape[0]}")
    return arr


# ─────────────────────────────────────────────
# ACTION DECODING / ENCODING
# ─────────────────────────────────────────────


def decode_action(
    obs_dict: Dict[str, Any],
    strategy_idx: int,
    action_type_idx: int,
    threat_idx: int,
    resource_idx: int,
    zone_idx: int,
    units_idx: int,
    severity_idx: int,
    tti_idx: int,
    pop_idx: int,
) -> Dict[str, Any]:
    mask = build_valid_action_mask(obs_dict)

    strategy = STRATEGY_TYPES[min(max(strategy_idx, 0), len(STRATEGY_TYPES) - 1)]
    action_type = ACTION_TYPES[min(max(action_type_idx, 0), len(ACTION_TYPES) - 1)]

    if mask["action_mask"][ACTION_TYPES.index(action_type)] == 0:
        action_type = "skip"

    threat_ids = mask["threat_ids"]
    resource_ids = mask["resource_ids"]
    zone_ids = mask["zone_ids"]

    selected_threat = threat_ids[min(threat_idx, len(threat_ids) - 1)] if threat_ids else None
    selected_resource = resource_ids[min(resource_idx, len(resource_ids) - 1)] if resource_ids else None
    selected_zone = zone_ids[min(zone_idx, len(zone_ids) - 1)] if zone_ids else None

    if action_type == "classify" and selected_threat is not None:
        threat = next((t for t in obs_dict.get("threats", []) if int(t.get("threat_id", -1)) == selected_threat), None)
        if threat is None:
            return {"action_type": "skip", "strategy": strategy}
        predicted_type = threat.get("threat_type", "airstrike")
        predicted_severity = float(np.clip((severity_idx / 9.0) * 10.0, 0.0, 10.0))
        return {
            "action_type": "classify",
            "strategy": strategy,
            "classification": {
                "threat_id": selected_threat,
                "predicted_type": predicted_type,
                "predicted_severity": round(predicted_severity, 2),
            },
        }

    if action_type == "predict" and selected_threat is not None:
        predicted_tti = int(np.clip((tti_idx / 19.0) * TOTAL_STEPS, 0, TOTAL_STEPS))
        predicted_pop = int(np.clip((pop_idx / 19.0) * 2400, 0, 2400))
        return {
            "action_type": "predict",
            "strategy": strategy,
            "prediction": {
                "threat_id": selected_threat,
                "predicted_tti": predicted_tti,
                "predicted_pop": predicted_pop,
            },
        }

    if action_type == "allocate" and selected_threat is not None and selected_resource is not None:
        return {
            "action_type": "allocate",
            "strategy": strategy,
            "allocation": {
                "threat_id": selected_threat,
                "resource_id": selected_resource,
            },
        }

    if action_type == "coordinate" and threat_ids:
        active = [t for t in obs_dict.get("threats", []) if t.get("status") == "active"]

        def priority(th: Dict[str, Any]) -> float:
            sev = float(th.get("severity", 0.0))
            pop = float(th.get("population_at_risk", 0.0))
            tti = max(float(th.get("time_to_impact", 1.0)), 1.0)
            unc = float(th.get("severity_uncertainty", 0.0)) + float(th.get("population_uncertainty", 0.0))
            base = (sev * pop) / tti
            if strategy == "predict_first":
                base *= (1.0 + unc * 0.25)
            return base

        ranked = sorted(active, key=priority, reverse=True)
        return {
            "action_type": "coordinate",
            "strategy": strategy,
            "coordination": {"priority_order": [int(t["threat_id"]) for t in ranked]},
        }

    if action_type == "rescue" and selected_zone is not None:
        units = int(np.clip(units_idx + 1, 1, MAX_RESCUE_UNITS))
        return {
            "action_type": "rescue",
            "strategy": strategy,
            "rescue": {
                "zone_id": selected_zone,
                "rescue_units_to_send": units,
            },
        }

    if action_type == "delay" and selected_threat is not None:
        delay_steps = int(np.clip((units_idx // 2) + 1, 1, 3))
        return {
            "action_type": "delay",
            "strategy": strategy,
            "delay": {
                "threat_id": selected_threat,
                "delay_steps": delay_steps,
            },
        }

    return {"action_type": "skip", "strategy": strategy}


def encode_action_labels(action_dict: Dict[str, Any], obs_dict: Dict[str, Any]) -> Dict[str, int]:
    """Map an action dict back to discrete label indices (used for behavior cloning)."""
    mask = build_valid_action_mask(obs_dict)

    strategy = action_dict.get("strategy", "balanced")
    action_type = action_dict.get("action_type", "skip")

    labels = {
        "strategy": STRATEGY_TYPES.index(strategy) if strategy in STRATEGY_TYPES else STRATEGY_TYPES.index("balanced"),
        "action_type": ACTION_TYPES.index(action_type) if action_type in ACTION_TYPES else ACTION_TYPES.index("skip"),
        "threat": 0,
        "resource": 0,
        "zone": 0,
        "units": 0,
        "severity": 5,
        "tti": 10,
        "pop": 10,
    }

    threat_ids = mask["threat_ids"]
    resource_ids = mask["resource_ids"]
    zone_ids = mask["zone_ids"]

    if action_type == "classify" and action_dict.get("classification"):
        payload = action_dict["classification"]
        tid = int(payload.get("threat_id", threat_ids[0] if threat_ids else 1))
        if tid in threat_ids:
            labels["threat"] = threat_ids.index(tid)
        labels["severity"] = int(np.clip(round(float(payload.get("predicted_severity", 5.0)) / 10.0 * 9), 0, 9))

    elif action_type == "predict" and action_dict.get("prediction"):
        payload = action_dict["prediction"]
        tid = int(payload.get("threat_id", threat_ids[0] if threat_ids else 1))
        if tid in threat_ids:
            labels["threat"] = threat_ids.index(tid)
        labels["tti"] = int(np.clip(round(float(payload.get("predicted_tti", 0)) / max(TOTAL_STEPS, 1) * 19), 0, 19))
        labels["pop"] = int(np.clip(round(float(payload.get("predicted_pop", 0)) / 2400.0 * 19), 0, 19))

    elif action_type == "allocate" and action_dict.get("allocation"):
        payload = action_dict["allocation"]
        tid = int(payload.get("threat_id", threat_ids[0] if threat_ids else 1))
        rid = int(payload.get("resource_id", resource_ids[0] if resource_ids else 1))
        if tid in threat_ids:
            labels["threat"] = threat_ids.index(tid)
        if rid in resource_ids:
            labels["resource"] = resource_ids.index(rid)

    elif action_type == "rescue" and action_dict.get("rescue"):
        payload = action_dict["rescue"]
        zid = int(payload.get("zone_id", zone_ids[0] if zone_ids else 1))
        units = int(payload.get("rescue_units_to_send", 1))
        if zid in zone_ids:
            labels["zone"] = zone_ids.index(zid)
        labels["units"] = int(np.clip(units - 1, 0, MAX_RESCUE_UNITS - 1))

    elif action_type == "delay" and action_dict.get("delay"):
        payload = action_dict["delay"]
        tid = int(payload.get("threat_id", threat_ids[0] if threat_ids else 1))
        if tid in threat_ids:
            labels["threat"] = threat_ids.index(tid)
        labels["units"] = int(np.clip((int(payload.get("delay_steps", 1)) - 1) * 2, 0, MAX_RESCUE_UNITS - 1))

    return labels


# ─────────────────────────────────────────────
# BASELINE AGENT (intentionally imperfect)
# ─────────────────────────────────────────────


def _priority_score(threat: Dict[str, Any]) -> float:
    severity = float(threat.get("severity", 0.0))
    population = int(threat.get("population_at_risk", 0))
    tti = max(int(threat.get("time_to_impact", 1)), 1)
    unc = float(threat.get("severity_uncertainty", 0.0)) + float(threat.get("population_uncertainty", 0.0))
    return (severity * population) / tti * (1.0 + unc * 0.10)


def _resource_match(threat: Dict[str, Any], resource: Dict[str, Any]) -> bool:
    affinity = ZONE_RESOURCE_AFFINITY.get(threat.get("zone"), [])
    return resource.get("resource_type") in affinity


def _ranked_resources(threat: Dict[str, Any], resources: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    available = [r for r in resources if r.get("is_available", False)]

    def score(r: Dict[str, Any]) -> float:
        bonus = 0.22 if _resource_match(threat, r) else 0.0
        return float(r.get("effectiveness", 0.0)) + bonus

    return sorted(available, key=score, reverse=True)


def baseline_classification(threat: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    sev_obs = float(threat.get("severity", 5.0))
    unc = float(threat.get("severity_uncertainty", 0.1))

    if rng.random() < 0.03 + unc * 0.05:
        wrong_types = [t for t in THREAT_TYPES if t != threat.get("threat_type")]
        predicted_type = rng.choice(wrong_types) if wrong_types else threat.get("threat_type", "airstrike")
    else:
        predicted_type = threat.get("threat_type", "airstrike")

    noisy_severity = float(np.clip(sev_obs + rng.uniform(-0.9, 0.9) * (0.45 + unc), 0.0, 10.0))

    return {
        "action_type": "classify",
        "strategy": "predict_first",
        "classification": {
            "threat_id": int(threat["threat_id"]),
            "predicted_type": predicted_type,
            "predicted_severity": round(noisy_severity, 2),
        },
    }


def baseline_prediction(threat: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    tti = max(int(threat.get("time_to_impact", 1)), 0)
    pop = max(int(threat.get("population_at_risk", 0)), 0)
    unc = float(threat.get("population_uncertainty", 0.1)) + float(threat.get("tti_uncertainty", 0.1))

    predicted_tti = max(0, int(round(tti + rng.choice([0, 0, 0, 1, -1]))))
    predicted_pop = max(0, int(round(pop * (0.98 + rng.uniform(-0.05, 0.05) + unc * 0.02))))

    return {
        "action_type": "predict",
        "strategy": "predict_first",
        "prediction": {
            "threat_id": int(threat["threat_id"]),
            "predicted_tti": predicted_tti,
            "predicted_pop": predicted_pop,
        },
    }


def baseline_coordinate(threats: Sequence[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    active = [t for t in threats if t.get("status") == "active"]
    ranked = sorted(active, key=_priority_score, reverse=True)
    ids = [int(t["threat_id"]) for t in ranked]

    # Intentional inefficiency: occasionally swap top-2 priorities.
    if len(ids) >= 2 and rng.random() < 0.28:
        ids[0], ids[1] = ids[1], ids[0]

    return {
        "action_type": "coordinate",
        "strategy": "balanced",
        "coordination": {"priority_order": ids},
    }


def baseline_allocate(threat: Dict[str, Any], resources: Sequence[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    ranked = _ranked_resources(threat, resources)
    if not ranked:
        return None

    # Imperfect choice: 25% chance to pick second-best resource.
    pick = ranked[0]
    if len(ranked) > 1 and rng.random() < 0.12:
        pick = ranked[1]

    return {
        "action_type": "allocate",
        "strategy": "rescue_first",
        "allocation": {
            "threat_id": int(threat["threat_id"]),
            "resource_id": int(pick["resource_id"]),
        },
    }


def baseline_rescue_action(zone: Dict[str, Any], budget_remaining: int, rng: random.Random) -> Optional[Dict[str, Any]]:
    remaining = int(zone.get("total_victims", 0)) - int(zone.get("rescued", 0))
    if not zone.get("is_active", False) or remaining <= 0 or budget_remaining <= 0:
        return None

    if remaining > 140:
        units = 5
    elif remaining > 70:
        units = 4
    else:
        units = 3

    # Slight inefficiency
    if rng.random() < 0.08:
        units = max(1, units - 1)

    units = min(units, MAX_RESCUE_UNITS, budget_remaining)
    if units <= 0:
        return None

    return {
        "action_type": "rescue",
        "strategy": "rescue_first",
        "rescue": {
            "zone_id": int(zone["zone_id"]),
            "rescue_units_to_send": int(units),
        },
    }


def baseline_delay(threat: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_type": "delay",
        "strategy": "balanced",
        "delay": {
            "threat_id": int(threat["threat_id"]),
            "delay_steps": 1,
        },
    }


def choose_baseline_action(
    observation: Dict[str, Any],
    rng: random.Random,
    classified: set,
    predicted: set,
    coordinated_step: int,
) -> Tuple[Dict[str, Any], int]:
    threats = observation.get("threats", [])
    zones = observation.get("affected_zones", [])
    resources = observation.get("resources", [])
    budget_remaining = int(observation.get("resource_budget_remaining", 0))
    step = int(observation.get("current_step", 0))

    active = [t for t in threats if t.get("status") == "active"]
    active_zones = [z for z in zones if z.get("is_active", False)]

    # Rescue-first when there are active victims.
    if active_zones and budget_remaining > 0:
        zone = max(active_zones, key=lambda z: int(z.get("total_victims", 0)) - int(z.get("rescued", 0)))
        action = baseline_rescue_action(zone, budget_remaining, rng)
        if action is not None:
            return action, coordinated_step

    # Periodic coordination (not every step).
    if active and (step - coordinated_step >= 2) and rng.random() < 0.90:
        return baseline_coordinate(threats, rng), step

    # Work on highest-priority threat.
    if active:
        target = sorted(active, key=_priority_score, reverse=True)[0]
        tid = int(target["threat_id"])

        if tid not in classified and rng.random() < 0.98:
            classified.add(tid)
            return baseline_classification(target, rng), coordinated_step

        if tid not in predicted and rng.random() < 0.96:
            predicted.add(tid)
            return baseline_prediction(target, rng), coordinated_step

        if budget_remaining > 0 and rng.random() < 0.90:
            alloc = baseline_allocate(target, resources, rng)
            if alloc is not None:
                return alloc, coordinated_step

        if int(target.get("time_to_impact", 99)) <= 2 and rng.random() < 0.55:
            return baseline_delay(target), coordinated_step

    # Intentional idle probability for weaker baseline.
    if rng.random() < 0.05:
        return {"action_type": "skip", "strategy": "balanced"}, coordinated_step

    # Fallback
    if active:
        return baseline_coordinate(threats, rng), step
    return {"action_type": "skip", "strategy": "balanced"}, coordinated_step


def run_local_baseline_episode(seed: int, difficulty: str = "medium") -> EpisodeSummary:
    env = CrisisEnvironment(seed=seed)
    observation = observation_to_dict(env.reset(seed=seed, difficulty=difficulty))

    rng = random.Random(seed + 911)
    total_reward = 0.0
    done = False

    classified: set[int] = set()
    predicted: set[int] = set()
    coordinated_step = -10

    while not done:
        action, coordinated_step = choose_baseline_action(
            observation=observation,
            rng=rng,
            classified=classified,
            predicted=predicted,
            coordinated_step=coordinated_step,
        )

        result = env.step(CrisisAction(**action))
        total_reward += float(result.reward)
        observation = observation_to_dict(result.observation)
        done = bool(result.done)

    state = env.state()
    task_scores = state_to_metrics(state)
    return EpisodeSummary(
        total_reward=round(total_reward, 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=int(state.step_count),
    )


def collect_baseline_dataset(
    episodes: int,
    seed: int,
    difficulty: str = "medium",
) -> List[Tuple[Dict[str, Any], Dict[str, int]]]:
    """
    Collect (observation, labels) pairs from baseline trajectories for behavior cloning.
    """
    dataset: List[Tuple[Dict[str, Any], Dict[str, int]]] = []

    for ep in range(episodes):
        ep_seed = seed + ep * 17
        env = CrisisEnvironment(seed=ep_seed)
        observation = observation_to_dict(env.reset(seed=ep_seed, difficulty=difficulty))

        rng = random.Random(ep_seed + 333)
        classified: set[int] = set()
        predicted: set[int] = set()
        coordinated_step = -10
        done = False

        while not done:
            action, coordinated_step = choose_baseline_action(
                observation=observation,
                rng=rng,
                classified=classified,
                predicted=predicted,
                coordinated_step=coordinated_step,
            )
            labels = encode_action_labels(action, observation)
            dataset.append((observation, labels))

            result = env.step(CrisisAction(**action))
            observation = observation_to_dict(result.observation)
            done = bool(result.done)

    return dataset
