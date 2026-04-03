"""
server/environment.py — Core simulation engine for the AI Crisis Response & Rescue Coordination Environment.

Design goals:
- True decision-making MDP/POMDP: any valid action at any step.
- Non-deterministic dynamics with seed-reproducibility.
- Explicit trade-offs: limited global resource budget and time pressure.
- Action masking support for efficient RL training.
- Reward aligned with final score to avoid objective mismatch.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import (
    ActionType,
    AffectedZoneInfo,
    AllocationPayload,
    ClassificationPayload,
    CoordinationPayload,
    CrisisAction,
    CrisisObservation,
    CrisisState,
    DelayPayload,
    EvacuationPayload,
    PredictionPayload,
    RescuePayload,
    ResourceInfo,
    ResourceType,
    StepResult,
    ThreatInfo,
    ThreatStatus,
    ThreatType,
    ZoneType,
)

# ─────────────────────────────────────────────
# CONSTANTS (publicly imported by utils)
# ─────────────────────────────────────────────

TOTAL_STEPS = 30
MAX_RESCUE_UNITS = 5
MAX_THREATS_VISIBLE = 6

# Limited global resource budget per episode (core trade-off)
GLOBAL_RESOURCE_BUDGET = 5

ZONE_RESOURCE_AFFINITY: Dict[ZoneType, List[ResourceType]] = {
    ZoneType.MILITARY: [ResourceType.MILITARY_UNIT, ResourceType.MEDICAL_TEAM],
    ZoneType.MARITIME: [ResourceType.COAST_GUARD, ResourceType.RESCUE_DRONE],
    ZoneType.URBAN: [ResourceType.SWAT_TEAM, ResourceType.FIRE_BRIGADE, ResourceType.EVACUATION_BUS],
    ZoneType.RURAL: [ResourceType.FIRE_BRIGADE, ResourceType.MEDICAL_TEAM, ResourceType.RESCUE_DRONE],
}

THREAT_TEMPLATES: List[Tuple[ThreatType, ZoneType, str, int]] = [
    (ThreatType.AIRSTRIKE, ZoneType.MILITARY, "Military Base Alpha", 70),
    (ThreatType.SHIP_ATTACK, ZoneType.MARITIME, "Naval Port Sector 7", 240),
    (ThreatType.DRONE_THREAT, ZoneType.URBAN, "Downtown Business District", 1200),
    (ThreatType.EXPLOSION, ZoneType.URBAN, "Central Train Station", 900),
    (ThreatType.FLOOD, ZoneType.RURAL, "River Valley District", 450),
    (ThreatType.FIRE, ZoneType.URBAN, "Industrial Complex East", 360),
]

RESOURCE_TEMPLATES: List[Tuple[ResourceType, ZoneType, float]] = [
    (ResourceType.MILITARY_UNIT, ZoneType.MILITARY, 0.90),
    (ResourceType.COAST_GUARD, ZoneType.MARITIME, 0.88),
    (ResourceType.SWAT_TEAM, ZoneType.URBAN, 0.85),
    (ResourceType.FIRE_BRIGADE, ZoneType.URBAN, 0.82),
    (ResourceType.MEDICAL_TEAM, ZoneType.MILITARY, 0.78),
    (ResourceType.RESCUE_DRONE, ZoneType.MARITIME, 0.72),
    (ResourceType.EVACUATION_BUS, ZoneType.URBAN, 0.66),
    (ResourceType.MEDICAL_TEAM, ZoneType.RURAL, 0.76),
]

ACTION_TYPE_ORDER: List[str] = [
    "classify",
    "predict",
    "allocate",
    "coordinate",
    "rescue",
    "skip",
    "delay",
]

STRATEGIES: List[str] = ["rescue_first", "predict_first", "balanced"]
CORE_ACTIONS: List[str] = ["classify", "predict", "allocate", "coordinate", "rescue"]
CORE_TASKS: List[str] = ["classification", "prediction", "allocation", "coordination", "rescue"]
TASK_IMPORTANCE: Dict[str, float] = {
    "classification": 1.0,
    "prediction": 1.0,
    "allocation": 1.6,
    "coordination": 1.5,
    "rescue": 1.8,
}

# Adaptive reward and regularization constants
ADAPTIVE_ALPHA = 3.0
ADAPTIVE_WEIGHT_FLOOR = 0.05
LAMBDA_TIME = 0.006
LAMBDA_RESOURCE_WASTE = 1.0
LAMBDA_PRIORITY_ERROR = 1.0
LAMBDA_DELAY = 1.0
COVERAGE_WINDOW = 10
SOFT_COVERAGE_BONUS = 0.02
DELAY_PROGRESS_K = 0.015
UNIFIED_REWARD_LAMBDA = 0.35
EXECUTION_ACTION_GAMMA = 0.30
TERMINAL_HIGH_RISK_BOOST = 0.10

# Empirical baseline action prior (used for KL regularization target).
DEFAULT_BASELINE_ACTION_DIST: Dict[str, float] = {
    "classify": 0.22,
    "predict": 0.21,
    "allocate": 0.22,
    "coordinate": 0.18,
    "rescue": 0.17,
}


@dataclass
class DifficultyProfile:
    name: str
    threats: int
    obs_noise: float
    escalation_prob: float
    spread_prob: float
    budget: int
    episode_steps: int


DIFFICULTY_PROFILES: Dict[str, DifficultyProfile] = {
    "easy": DifficultyProfile("easy", threats=2, obs_noise=0.08, escalation_prob=0.08, spread_prob=0.05, budget=6, episode_steps=22),
    "medium": DifficultyProfile("medium", threats=3, obs_noise=0.16, escalation_prob=0.15, spread_prob=0.12, budget=5, episode_steps=26),
    "hard": DifficultyProfile("hard", threats=4, obs_noise=0.24, escalation_prob=0.23, spread_prob=0.20, budget=4, episode_steps=30),
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────


class CrisisEnvironment:
    """
    Crisis response environment with partial observability and stochastic dynamics.

    Compatible API:
      - reset(seed?, difficulty?) -> CrisisObservation
      - step(action) -> StepResult
      - state() -> CrisisState
      - task_scores() -> Dict[str, float]
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed if seed is not None else 42
        self._rng = random.Random(self._seed)

        self._difficulty = "medium"
        self._profile = DIFFICULTY_PROFILES[self._difficulty]
        self._episode_total_steps = self._profile.episode_steps

        self._episode_id = ""
        self._step_count = 0
        self._done = False

        self._threats: List[ThreatInfo] = []
        self._resources: List[ResourceInfo] = []
        self._affected_zones: List[AffectedZoneInfo] = []

        # Hidden truth per threat for partial observability
        self._true_state: Dict[int, Dict[str, float]] = {}

        # Dynamics state
        self._delay_buffer: Dict[int, int] = {}
        self._threat_mitigation: Dict[int, float] = {}

        # Resource and budget constraints
        self._resource_budget_total = GLOBAL_RESOURCE_BUDGET
        self._resource_budget_remaining = GLOBAL_RESOURCE_BUDGET
        self._resource_spent = 0
        self._wasted_resource_events = 0

        # Metrics
        self._total_population = 0
        self._casualties = 0
        self._casualties_prevented = 0
        self._cumulative_reward = 0.0

        # Task accumulators
        self._classify_scores: Dict[int, float] = {}
        self._predict_scores: Dict[int, float] = {}
        self._alloc_scores: List[float] = []
        self._coord_scores: List[float] = []
        self._rescue_total_victims = 0
        self._rescue_saved = 0
        self._rescue_steps: List[int] = []

        # Penalty trackers used in scoring
        self._wrong_priority_events = 0
        self._critical_ignored_steps = 0

        # Short memory exposed in observation
        self._recent_actions: List[str] = []
        self._action_counts: Dict[str, int] = {a: 0 for a in CORE_ACTIONS}
        self._last_action_step: Dict[str, int] = {a: -999 for a in CORE_ACTIONS}
        self._baseline_action_dist: Dict[str, float] = dict(DEFAULT_BASELINE_ACTION_DIST)

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, difficulty: str = "medium", **kwargs) -> CrisisObservation:
        if seed is not None:
            self._seed = seed
        if self._seed is None:
            self._seed = 42
        self._rng = random.Random(self._seed)

        self._difficulty = difficulty if difficulty in DIFFICULTY_PROFILES else "medium"
        self._profile = DIFFICULTY_PROFILES[self._difficulty]
        self._episode_total_steps = self._profile.episode_steps

        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._done = False

        self._threats = self._generate_threats(self._profile.threats)
        self._resources = self._generate_resources()
        self._affected_zones = []

        self._true_state = {
            t.threat_id: {
                "severity": float(t.severity),
                "population": float(t.population_at_risk),
                "tti": float(t.time_to_impact),
                "initial_tti": float(t.time_to_impact),
            }
            for t in self._threats
        }
        self._delay_buffer = {t.threat_id: 0 for t in self._threats}
        self._threat_mitigation = {t.threat_id: 0.0 for t in self._threats}

        self._resource_budget_total = self._profile.budget
        self._resource_budget_remaining = self._profile.budget
        self._resource_spent = 0
        self._wasted_resource_events = 0

        self._total_population = sum(int(v["population"]) for v in self._true_state.values())
        self._casualties = 0
        self._casualties_prevented = 0
        self._cumulative_reward = 0.0

        self._classify_scores = {}
        self._predict_scores = {}
        self._alloc_scores = []
        self._coord_scores = []
        self._rescue_total_victims = 0
        self._rescue_saved = 0
        self._rescue_steps = []
        self._wrong_priority_events = 0
        self._critical_ignored_steps = 0
        self._recent_actions = []
        self._action_counts = {a: 0 for a in CORE_ACTIONS}
        self._last_action_step = {a: -999 for a in CORE_ACTIONS}

        alerts = [
            f"[EPISODE {self._episode_id}] INITIATED — difficulty={self._difficulty}.",
            f"Threats={len(self._threats)} | Global resource budget={self._resource_budget_total}",
            "State is partially observable: threat attributes are noisy estimates.",
        ]
        return self._build_observation(alerts)

    def step(self, action: CrisisAction) -> StepResult:
        if self._done:
            obs = self._build_observation(["Episode already ended. Call reset()."])
            return StepResult(observation=obs, reward=0.0, done=True, info={})

        prev_task_scores = self.task_scores()
        action_name = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        prev_last_action_step = int(self._last_action_step.get(action_name, -999))

        task_success_reward = 0.0
        time_penalty = 1.0
        resource_waste_penalty = 0.0
        wrong_priority_penalty = 0.0
        delay_penalty = 0.0

        alerts: List[str] = []
        info: Dict[str, float] = {}

        action_bonus, action_alerts, action_info = self._process_action(action)
        alerts.extend(action_alerts)
        info.update(action_info)

        resource_waste_penalty += float(action_info.get("resource_waste_penalty", 0.0))
        wrong_priority_penalty += float(action_info.get("wrong_priority_penalty", 0.0))

        lifecycle_alerts, lifecycle_info = self._advance_dynamics()
        alerts.extend(lifecycle_alerts)

        resource_waste_penalty += float(lifecycle_info.get("resource_waste_penalty", 0.0))
        wrong_priority_penalty += float(lifecycle_info.get("wrong_priority_penalty", 0.0))
        delay_penalty += float(lifecycle_info.get("delay_penalty", 0.0))

        self._step_count += 1

        no_active_threats = not any(t.status == ThreatStatus.ACTIVE for t in self._threats)
        no_active_zones = not any(z.is_active for z in self._affected_zones)
        time_up = self._step_count >= self._episode_total_steps

        if (no_active_threats and no_active_zones) or time_up:
            self._done = True

        current_task_scores = self.task_scores()
        adaptive_weights = self._adaptive_task_weights(prev_task_scores)
        weighted_task_progress = sum(
            adaptive_weights[k] * (current_task_scores[k] - prev_task_scores[k])
            for k in CORE_TASKS
        )
        task_success_reward = weighted_task_progress + action_bonus

        # Time-efficiency pressure grows with elapsed steps.
        delay_progress_penalty = DELAY_PROGRESS_K * float(self._step_count)
        raw_step_reward = (
            task_success_reward
            - LAMBDA_TIME * time_penalty
            - LAMBDA_RESOURCE_WASTE * resource_waste_penalty
            - LAMBDA_PRIORITY_ERROR * wrong_priority_penalty
            - LAMBDA_DELAY * delay_penalty
            - delay_progress_penalty
        )

        if action_name in {"allocate", "coordinate", "rescue"}:
            raw_step_reward *= 1.0 + EXECUTION_ACTION_GAMMA

        coverage_bonus = self._action_coverage_bonus(action_name, prev_last_action_step)
        raw_step_reward += coverage_bonus

        terminal_reward = 0.0
        high_risk_terminal_boost = 0.0
        if self._done:
            terminal_reward = _clamp(self._rescue_saved / max(self._total_population, 1))
            high_risk_terminal_boost = self._terminal_high_risk_boost()
            terminal_reward += high_risk_terminal_boost

        raw_reward = (1.0 - UNIFIED_REWARD_LAMBDA) * raw_step_reward + UNIFIED_REWARD_LAMBDA * terminal_reward
        step_reward = self._normalize_step_reward(raw_reward)

        self._cumulative_reward += step_reward

        info.update(
            {
                "task_success_reward": round(task_success_reward, 6),
                "time_penalty": round(time_penalty, 6),
                "resource_waste_penalty": round(resource_waste_penalty, 6),
                "wrong_priority_penalty": round(wrong_priority_penalty, 6),
                "delay_penalty": round(delay_penalty, 6),
                "delay_progress_penalty": round(delay_progress_penalty, 6),
                "balance_kl": 0.0,
                "coverage_penalty": 0.0,
                "coverage_bonus": round(coverage_bonus, 6),
                "execution_multiplier": round((1.0 + EXECUTION_ACTION_GAMMA) if action_name in {"allocate", "coordinate", "rescue"} else 1.0, 6),
                "sequence_bonus": 0.0,
                "rescue_priority_bonus": 0.0,
                "terminal_rescue_boost": round(terminal_reward, 6),
                "terminal_high_risk_boost": round(high_risk_terminal_boost, 6),
                "action_bonus": round(action_bonus, 6),
                "adaptive_weights": {k: round(v, 5) for k, v in adaptive_weights.items()},
                "raw_step_reward": round(raw_step_reward, 6),
                "raw_reward": round(raw_reward, 6),
            }
        )

        obs = self._build_observation(alerts)
        return StepResult(observation=obs, reward=round(step_reward, 6), done=self._done, info=info)

    def state(self) -> CrisisState:
        c_score = self._grader_classification()
        p_score = self._grader_prediction()
        a_score = self._grader_allocation()
        co_score = self._grader_coordination()
        r_score = self._grader_rescue()
        final = self._compute_final_score(
            cached=(c_score, p_score, a_score, co_score, r_score)
        )

        total_victims = max(self._rescue_total_victims, 1)
        rescue_rate = _clamp(self._rescue_saved / total_victims)

        resolved = sum(1 for t in self._threats if t.status in {ThreatStatus.CONTAINED, ThreatStatus.RESOLVED})

        return CrisisState(
            step_count=self._step_count,
            total_steps=self._episode_total_steps,
            episode_id=self._episode_id,
            difficulty=self._difficulty,
            classification_score=c_score,
            prediction_score=p_score,
            allocation_score=a_score,
            coordination_score=co_score,
            rescue_score=r_score,
            final_score=final,
            resolved_threats=resolved,
            total_threats=len(self._threats),
            casualties=self._casualties,
            casualties_prevented=self._casualties_prevented,
            total_population_at_risk=self._total_population,
            rescue_success_rate=round(rescue_rate, 4),
            resource_budget_remaining=self._resource_budget_remaining,
            resource_budget_total=self._resource_budget_total,
            cumulative_reward=round(self._cumulative_reward, 6),
            done=self._done,
        )

    def task_scores(self) -> Dict[str, float]:
        return {
            "classification": self._grader_classification(),
            "prediction": self._grader_prediction(),
            "allocation": self._grader_allocation(),
            "coordination": self._grader_coordination(),
            "rescue": self._grader_rescue(),
        }

    def set_baseline_action_distribution(self, action_dist: Dict[str, float]) -> None:
        """
        Optional trainer hook for setting empirical baseline action frequencies.
        Does not change external OpenEnv interface.
        """
        cleaned = {
            a: max(float(action_dist.get(a, 0.0)), 0.0)
            for a in CORE_ACTIONS
        }
        total = sum(cleaned.values())
        if total <= 0:
            self._baseline_action_dist = dict(DEFAULT_BASELINE_ACTION_DIST)
            return
        self._baseline_action_dist = {a: cleaned[a] / total for a in CORE_ACTIONS}

    def valid_actions(self) -> Dict[str, object]:
        active_threats = sorted([t for t in self._threats if t.status == ThreatStatus.ACTIVE], key=lambda t: t.threat_id)
        active_zones = sorted([z for z in self._affected_zones if z.is_active], key=lambda z: z.zone_id)
        available_resources = sorted([r for r in self._resources if r.is_available], key=lambda r: r.resource_id)

        action_mask = {
            "classify": int(bool(active_threats)),
            "predict": int(bool(active_threats)),
            "allocate": int(bool(active_threats and available_resources and self._resource_budget_remaining > 0)),
            "coordinate": int(len(active_threats) >= 2),
            "rescue": int(bool(active_zones and self._resource_budget_remaining > 0)),
            "skip": 1,
            "delay": int(bool(active_threats)),
        }

        return {
            "action_types": list(ACTION_TYPE_ORDER),
            "action_mask": [action_mask[a] for a in ACTION_TYPE_ORDER],
            "threat_ids": [t.threat_id for t in active_threats],
            "resource_ids": [r.resource_id for r in available_resources],
            "zone_ids": [z.zone_id for z in active_zones],
            "max_rescue_units": min(MAX_RESCUE_UNITS, max(0, self._resource_budget_remaining)),
            "strategy_options": list(STRATEGIES),
        }

    # ─────────────────────────────────────────
    # ACTION HANDLERS
    # ─────────────────────────────────────────

    def _process_action(self, action: CrisisAction) -> Tuple[float, List[str], Dict[str, float]]:
        self._record_action_memory(action)

        if action.action_type == ActionType.CLASSIFY and action.classification:
            return self._handle_classify(action.classification)
        if action.action_type == ActionType.PREDICT and action.prediction:
            return self._handle_predict(action.prediction)
        if action.action_type == ActionType.ALLOCATE and action.allocation:
            return self._handle_allocate(action.allocation)
        if action.action_type == ActionType.COORDINATE and action.coordination:
            return self._handle_coordinate(action.coordination)
        if action.action_type == ActionType.RESCUE and action.rescue:
            return self._handle_rescue(action.rescue)
        if action.action_type == ActionType.DELAY and action.delay:
            return self._handle_delay(action.delay)
        if action.action_type == ActionType.SKIP:
            return 0.0, ["[SKIP] Holding action this step."], {}
        if action.action_type == ActionType.EVACUATE and action.evacuate:
            return self._handle_evacuate(action.evacuate)

        return -0.02, ["[WARN] Invalid or incomplete action payload."], {"resource_waste_penalty": 0.01}

    def _handle_classify(self, payload: ClassificationPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat = self._get_threat(payload.threat_id)
        if threat is None or threat.status != ThreatStatus.ACTIVE:
            return -0.02, ["[WARN] CLASSIFY target invalid or not active."], {"resource_waste_penalty": 0.01}

        truth = self._true_state[threat.threat_id]
        type_correct = payload.predicted_type == threat.threat_type
        sev_error = abs(float(payload.predicted_severity) - float(truth["severity"]))
        tolerance = 1.0 + self._profile.obs_noise * 2.5

        if type_correct and sev_error <= tolerance:
            score = 1.0
            label = "CORRECT"
        elif type_correct:
            score = 0.55
            label = "PARTIAL"
        else:
            score = 0.0
            label = "WRONG"

        self._classify_scores[threat.threat_id] = max(self._classify_scores.get(threat.threat_id, 0.0), score)
        threat.predicted_severity = float(payload.predicted_severity)

        bonus = score * 0.04
        return bonus, [
            f"[CLASSIFY] Threat {threat.threat_id} -> {label} (sev_err={sev_error:.2f})"
        ], {}

    def _handle_predict(self, payload: PredictionPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat = self._get_threat(payload.threat_id)
        if threat is None or threat.status != ThreatStatus.ACTIVE:
            return -0.02, ["[WARN] PREDICT target invalid or not active."], {"resource_waste_penalty": 0.01}

        truth = self._true_state[threat.threat_id]
        tti_error = abs(float(payload.predicted_tti) - float(truth["tti"]))
        pop_error = abs(float(payload.predicted_pop) - float(truth["population"])) / max(float(truth["population"]), 1.0)

        norm_tti_error = _clamp(tti_error / max(self._episode_total_steps, 1), 0.0, 1.0)
        error = 0.5 * norm_tti_error + 0.5 * _clamp(pop_error, 0.0, 1.0)
        score = _clamp(1.0 - error)

        self._predict_scores[threat.threat_id] = max(self._predict_scores.get(threat.threat_id, 0.0), score)
        threat.predicted_tti = int(payload.predicted_tti)
        threat.predicted_pop = int(payload.predicted_pop)

        bonus = score * 0.04
        return bonus, [
            f"[PREDICT] Threat {threat.threat_id} -> quality={score:.3f}"
        ], {}

    def _handle_allocate(self, payload: AllocationPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat = self._get_threat(payload.threat_id)
        resource = self._get_resource(payload.resource_id)

        if threat is None or threat.status != ThreatStatus.ACTIVE or resource is None or not resource.is_available:
            return -0.03, ["[WARN] ALLOCATE invalid threat/resource."], {"resource_waste_penalty": 0.02}

        if self._resource_budget_remaining <= 0:
            self._wasted_resource_events += 1
            return -0.04, ["[WARN] ALLOCATE blocked: resource budget exhausted."], {"resource_waste_penalty": 0.04}

        self._spend_budget(1)
        resource.is_available = False
        resource.assigned_to = threat.threat_id
        resource.cooldown_steps = self._rng.randint(1, 3)
        threat.assigned_resource = resource.resource_id

        affinity = ZONE_RESOURCE_AFFINITY.get(threat.zone, [])
        match_bonus = 0.18 if resource.resource_type in affinity else -0.08

        effectiveness_noise = self._rng.uniform(-0.12, 0.12)
        effective_power = _clamp(resource.effectiveness + match_bonus + effectiveness_noise)
        self._threat_mitigation[threat.threat_id] = _clamp(
            self._threat_mitigation.get(threat.threat_id, 0.0) + effective_power * 0.55,
            0.0,
            0.95,
        )

        alloc_score = _clamp(resource.effectiveness + (0.2 if resource.resource_type in affinity else 0.0))
        self._alloc_scores.append(alloc_score)

        wrong_priority_penalty = 0.0
        highest = self._highest_risk_active_threat()
        if highest is not None and highest.threat_id != threat.threat_id:
            chosen_risk = self._true_priority(threat)
            highest_risk = self._true_priority(highest)
            if highest_risk > chosen_risk * 1.35:
                wrong_priority_penalty = 0.02
                self._wrong_priority_events += 1

        resource_waste_penalty = 0.0 if alloc_score >= 0.72 else 0.015
        if resource_waste_penalty > 0:
            self._wasted_resource_events += 1

        bonus = alloc_score * 0.05
        return bonus, [
            f"[ALLOCATE] Resource {resource.resource_id} -> Threat {threat.threat_id} | quality={alloc_score:.3f}"
        ], {
            "resource_waste_penalty": resource_waste_penalty,
            "wrong_priority_penalty": wrong_priority_penalty,
        }

    def _handle_coordinate(self, payload: CoordinationPayload) -> Tuple[float, List[str], Dict[str, float]]:
        active = sorted([t for t in self._threats if t.status == ThreatStatus.ACTIVE], key=lambda t: t.threat_id)
        active_ids = [t.threat_id for t in active]
        if len(active_ids) < 2:
            return -0.01, ["[WARN] COORDINATE ignored: less than 2 active threats."], {"resource_waste_penalty": 0.005}

        order = [tid for tid in payload.priority_order if tid in active_ids]
        if not order:
            return -0.02, ["[WARN] COORDINATE ignored: no valid threat IDs."], {"resource_waste_penalty": 0.01}

        ideal = [t.threat_id for t in sorted(active, key=self._true_priority, reverse=True)]
        score = self._rank_correlation_score(order, ideal)
        self._coord_scores.append(score)

        # Mark rank on threats for interpretability.
        for rank, tid in enumerate(order):
            t = self._get_threat(tid)
            if t is not None:
                t.priority_rank = rank + 1

        wrong_priority_penalty = 0.0
        if order[0] != ideal[0]:
            wrong_priority_penalty = 0.015
            self._wrong_priority_events += 1

        bonus = score * 0.05
        return bonus, [
            f"[COORDINATE] priority={order} | score={score:.3f}"
        ], {"wrong_priority_penalty": wrong_priority_penalty}

    def _handle_rescue(self, payload: RescuePayload) -> Tuple[float, List[str], Dict[str, float]]:
        zone = self._get_zone(payload.zone_id)
        if zone is None or not zone.is_active:
            return -0.02, ["[WARN] RESCUE target invalid or inactive."], {"resource_waste_penalty": 0.01}

        remaining = max(zone.total_victims - zone.rescued, 0)
        if remaining <= 0:
            zone.is_active = False
            return 0.0, [f"[RESCUE] Zone {zone.zone_id} already cleared."], {}

        requested = max(1, int(payload.rescue_units_to_send))
        spendable = min(requested, MAX_RESCUE_UNITS, self._resource_budget_remaining)
        if spendable <= 0:
            self._wasted_resource_events += 1
            return -0.04, ["[WARN] RESCUE blocked: resource budget exhausted."], {"resource_waste_penalty": 0.04}

        self._spend_budget(spendable)
        zone.rescue_units_deployed += spendable

        zone_multiplier = {
            ZoneType.URBAN: 1.0,
            ZoneType.MARITIME: 0.85,
            ZoneType.MILITARY: 0.9,
            ZoneType.RURAL: 0.78,
        }.get(zone.zone_type, 1.0)

        # Stochastic rescue effectiveness (seeded dynamics)
        rescue_effectiveness = _clamp(self._rng.uniform(0.76, 1.08) - self._profile.obs_noise * 0.2, 0.45, 1.15)
        saved = min(remaining, int(spendable * 14 * zone_multiplier * rescue_effectiveness))
        zone.rescued += saved
        self._rescue_saved += saved
        self._rescue_steps.append(self._step_count)

        if zone.rescued >= zone.total_victims:
            zone.is_active = False

        resource_waste_penalty = 0.0
        if requested > spendable:
            resource_waste_penalty += 0.01
            self._wasted_resource_events += 1
        if saved < max(1, int(spendable * 7)):
            resource_waste_penalty += 0.01
            self._wasted_resource_events += 1

        speed_factor = _clamp(1.0 - (self._step_count / max(self._episode_total_steps, 1)), 0.0, 1.0)
        bonus = (_clamp(saved / max(zone.total_victims, 1)) * 0.06) + speed_factor * 0.01
        high_priority = self._is_high_priority_rescue_target(zone.zone_id)

        return bonus, [
            f"[RESCUE] Zone {zone.zone_id}: saved={saved}, remaining={zone.total_victims - zone.rescued}"
        ], {
            "resource_waste_penalty": resource_waste_penalty,
            "high_priority_rescue": 1.0 if high_priority else 0.0,
        }

    def _handle_delay(self, payload: DelayPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat = self._get_threat(payload.threat_id)
        if threat is None or threat.status != ThreatStatus.ACTIVE:
            return -0.02, ["[WARN] DELAY target invalid or not active."], {"resource_waste_penalty": 0.01}

        requested = int(payload.delay_steps)
        success_prob = _clamp(0.62 - self._profile.obs_noise * 0.7 + self._threat_mitigation.get(threat.threat_id, 0.0) * 0.2, 0.2, 0.85)

        if self._rng.random() < success_prob:
            self._delay_buffer[threat.threat_id] = min(3, self._delay_buffer.get(threat.threat_id, 0) + requested)
            bonus = 0.012 * requested
            return bonus, [f"[DELAY] Threat {threat.threat_id} delayed by {requested} step(s)."], {}

        # Failed delay attempt can worsen dynamics under uncertainty.
        truth = self._true_state[threat.threat_id]
        truth["severity"] = min(10.0, truth["severity"] + self._rng.uniform(0.1, 0.5))
        truth["population"] = truth["population"] * self._rng.uniform(1.01, 1.06)
        return -0.01, [f"[DELAY] Threat {threat.threat_id} delay failed; escalation observed."], {"wrong_priority_penalty": 0.008}

    def _handle_evacuate(self, payload: EvacuationPayload) -> Tuple[float, List[str], Dict[str, float]]:
        """Retained for compatibility. Treated as pre-impact population reduction action."""
        threat = self._get_threat(payload.zone_id)
        if threat is None or threat.status != ThreatStatus.ACTIVE:
            return -0.02, ["[WARN] EVACUATE target invalid."], {"resource_waste_penalty": 0.01}
        if self._resource_budget_remaining <= 0:
            return -0.04, ["[WARN] EVACUATE blocked: resource budget exhausted."], {"resource_waste_penalty": 0.04}

        units = min(max(1, int(payload.evac_units)), self._resource_budget_remaining)
        self._spend_budget(units)

        truth = self._true_state[threat.threat_id]
        moved = int(min(truth["population"], units * 20 * self._rng.uniform(0.7, 1.1)))
        truth["population"] = max(0.0, truth["population"] - moved)
        threat.population_evacuated += moved

        bonus = _clamp(moved / max(self._total_population, 1), 0.0, 0.05)
        return bonus, [f"[EVACUATE] Threat {threat.threat_id}: evacuated={moved}"], {}

    # ─────────────────────────────────────────
    # DYNAMICS
    # ─────────────────────────────────────────

    def _advance_dynamics(self) -> Tuple[List[str], Dict[str, float]]:
        alerts: List[str] = []
        info: Dict[str, float] = {
            "resource_waste_penalty": 0.0,
            "wrong_priority_penalty": 0.0,
            "delay_penalty": 0.0,
        }

        # Cool down resources and release when ready.
        for resource in self._resources:
            if resource.cooldown_steps > 0:
                resource.cooldown_steps -= 1
                if resource.cooldown_steps == 0:
                    resource.is_available = True
                    resource.assigned_to = None

        active_threats = [t for t in self._threats if t.status == ThreatStatus.ACTIVE]

        for threat in active_threats:
            tid = threat.threat_id
            truth = self._true_state[tid]
            mitigation = self._threat_mitigation.get(tid, 0.0)

            # Escalation stochasticity
            escalation_prob = _clamp(self._profile.escalation_prob * (1.1 - mitigation), 0.02, 0.45)
            if self._rng.random() < escalation_prob:
                truth["severity"] = min(10.0, truth["severity"] + self._rng.uniform(0.2, 0.8))
                truth["population"] *= self._rng.uniform(1.03, 1.10)
                alerts.append(f"[ESCALATION] Threat {tid} intensified.")

            # Secondary spread (new threat) under high pressure
            if (
                len(self._threats) < MAX_THREATS_VISIBLE
                and truth["severity"] >= 7.0
                and self._rng.random() < self._profile.spread_prob * (1.0 - mitigation)
            ):
                new = self._spawn_secondary_threat(parent=threat)
                alerts.append(f"[SPREAD] Secondary threat {new.threat_id} emerged near {threat.location_name}.")

            # Delay consumes one decay step if available.
            if self._delay_buffer.get(tid, 0) > 0:
                self._delay_buffer[tid] -= 1
            else:
                truth["tti"] = max(0.0, truth["tti"] - 1.0)

            # Natural mitigation decay.
            self._threat_mitigation[tid] = _clamp(mitigation * 0.92, 0.0, 0.95)

            # Optional containment chance when heavily mitigated.
            if truth["tti"] > 0 and self._threat_mitigation[tid] > 0.72 and self._rng.random() < 0.35:
                threat.status = ThreatStatus.CONTAINED
                prevented = int(truth["population"] * 0.55)
                self._casualties_prevented += prevented
                threat.casualties_prevented += prevented
                alerts.append(f"[CONTAINED] Threat {tid} neutralized before impact.")
                continue

            if truth["tti"] <= 0.0:
                casualties, prevented = self._compute_impact(threat)
                threat.status = ThreatStatus.IMPACTED
                threat.casualties = casualties
                threat.casualties_prevented = prevented

                self._casualties += casualties
                self._casualties_prevented += prevented

                zone = AffectedZoneInfo(
                    zone_id=threat.threat_id,
                    zone_type=threat.zone,
                    location_name=threat.location_name,
                    total_victims=casualties,
                    is_active=casualties > 0,
                )
                self._affected_zones.append(zone)
                self._rescue_total_victims += max(0, casualties)

                alerts.append(
                    f"[IMPACT] Threat {tid} impacted at {threat.location_name}. victims={casualties}, prevented={prevented}"
                )

        # If zones are fully rescued, mark impacted threat resolved.
        for zone in self._affected_zones:
            if zone.is_active:
                continue
            threat = self._get_threat(zone.zone_id)
            if threat and threat.status == ThreatStatus.IMPACTED and zone.rescued >= zone.total_victims:
                threat.status = ThreatStatus.RESOLVED
                alerts.append(f"[RESOLVED] Threat {threat.threat_id} resolved after rescue.")

        # Delay penalty for ignoring urgent critical threats.
        critical = [t for t in self._threats if t.status == ThreatStatus.ACTIVE and self._is_critical(t)]
        unhandled_critical = [t for t in critical if t.assigned_resource is None and self._delay_buffer.get(t.threat_id, 0) == 0]
        if unhandled_critical:
            penalty = 0.006 * len(unhandled_critical)
            info["delay_penalty"] += penalty
            self._critical_ignored_steps += len(unhandled_critical)

        return alerts, info

    def _compute_impact(self, threat: ThreatInfo) -> Tuple[int, int]:
        truth = self._true_state[threat.threat_id]
        severity = _clamp(truth["severity"] / 10.0)
        population = max(0, int(truth["population"]))

        mitigation = self._threat_mitigation.get(threat.threat_id, 0.0)
        prediction_support = self._predict_scores.get(threat.threat_id, 0.0) * 0.08
        classification_support = self._classify_scores.get(threat.threat_id, 0.0) * 0.05
        total_mitigation = _clamp(mitigation + prediction_support + classification_support, 0.0, 0.92)

        base_rate = _clamp(0.08 + severity * 0.72, 0.05, 0.95)
        chaos_factor = self._rng.uniform(0.85, 1.15)
        casualty_rate = _clamp(base_rate * chaos_factor * (1.0 - total_mitigation), 0.01, 0.98)

        casualties = int(population * casualty_rate)
        prevented = max(0, population - casualties)
        return casualties, prevented

    # ─────────────────────────────────────────
    # GRADERS / FINAL SCORE
    # ─────────────────────────────────────────

    def _grader_classification(self) -> float:
        active_and_impacted = [t for t in self._threats if t.status in {ThreatStatus.ACTIVE, ThreatStatus.IMPACTED, ThreatStatus.RESOLVED, ThreatStatus.CONTAINED}]
        if not active_and_impacted:
            return 0.0
        scores = [self._classify_scores.get(t.threat_id, 0.0) for t in active_and_impacted]
        return round(_clamp(_mean(scores)), 4)

    def _grader_prediction(self) -> float:
        relevant = [t for t in self._threats if t.status in {ThreatStatus.ACTIVE, ThreatStatus.IMPACTED, ThreatStatus.RESOLVED, ThreatStatus.CONTAINED}]
        if not relevant:
            return 0.0
        scores = [self._predict_scores.get(t.threat_id, 0.0) for t in relevant]
        return round(_clamp(_mean(scores)), 4)

    def _grader_allocation(self) -> float:
        if not self._alloc_scores:
            return 0.0
        base = _mean(self._alloc_scores)
        waste_penalty = 0.03 * min(self._wasted_resource_events, 8) / 8.0
        budget_efficiency = 1.0 - (_clamp(self._resource_spent / max(self._resource_budget_total, 1), 0.0, 1.5) - 1.0 if self._resource_spent > self._resource_budget_total else 0.0)
        score = _clamp(base * 0.82 + _clamp(budget_efficiency) * 0.18 - waste_penalty)
        return round(score, 4)

    def _grader_coordination(self) -> float:
        if not self._coord_scores:
            return 0.0
        base = _mean(self._coord_scores)
        wrong_penalty = 0.04 * min(self._wrong_priority_events, 6) / 6.0
        return round(_clamp(base - wrong_penalty), 4)

    def _grader_rescue(self) -> float:
        if self._rescue_total_victims <= 0:
            # Threats prevented before impact is good rescue preparedness.
            preventive = _clamp(self._casualties_prevented / max(self._total_population, 1))
            return round(_clamp(0.55 + preventive * 0.45), 4)

        saved_ratio = _clamp(self._rescue_saved / max(self._rescue_total_victims, 1))
        if self._rescue_steps:
            avg_step = _mean([float(s) for s in self._rescue_steps])
            speed_score = _clamp(1.0 - avg_step / max(self._episode_total_steps, 1))
        else:
            speed_score = 0.0

        deployed = sum(z.rescue_units_deployed for z in self._affected_zones)
        efficiency = _clamp(self._rescue_saved / max(1, deployed * 14))
        score = _clamp(saved_ratio * 0.55 + speed_score * 0.25 + efficiency * 0.20)
        return round(score, 4)

    def _compute_final_score(self, cached: Optional[Tuple[float, float, float, float, float]] = None) -> float:
        if cached is None:
            c = self._grader_classification()
            p = self._grader_prediction()
            a = self._grader_allocation()
            co = self._grader_coordination()
            r = self._grader_rescue()
        else:
            c, p, a, co, r = cached

        weighted = (
            c * 0.30
            + p * 0.28
            + a * 0.14
            + co * 0.10
            + r * 0.18
        )

        # Explicitly include constraints in final score to keep reward/objective aligned.
        budget_eff = _clamp(1.0 - (self._resource_spent / max(self._resource_budget_total, 1) - 1.0) * 0.5 if self._resource_spent > self._resource_budget_total else 1.0)
        time_eff = _clamp(1.0 - self._step_count / max(self._episode_total_steps, 1))
        critical_ignore_penalty = _clamp(self._critical_ignored_steps / max(self._episode_total_steps * 3, 1))

        # Weighted task aggregation emphasizes uncertain reasoning tasks
        # (classification + prediction) while preserving operational objectives.
        score = 0.22 + weighted * 0.95 + budget_eff * 0.05 + time_eff * 0.04 - critical_ignore_penalty * 0.04
        return round(_clamp(score, 0.0, 1.0), 4)

    # ─────────────────────────────────────────
    # OBSERVATION / PARTIAL OBSERVABILITY
    # ─────────────────────────────────────────

    def _build_observation(self, alerts: List[str]) -> CrisisObservation:
        visible_threats: List[ThreatInfo] = []
        for threat in self._threats:
            truth = self._true_state[threat.threat_id]
            obs = threat.model_copy(deep=True)

            # Deterministic per-step observation noise so repeated reads don't alter dynamics.
            sev_noise = self._stable_noise(threat.threat_id, 11, self._profile.obs_noise * 2.2)
            pop_noise = self._stable_noise(threat.threat_id, 17, max(20.0, truth["population"] * self._profile.obs_noise * 0.18))
            tti_noise = self._stable_noise(threat.threat_id, 23, max(0.4, self._profile.obs_noise * 2.5))

            sev_obs = _clamp(truth["severity"] + sev_noise, 0.0, 10.0)
            pop_obs = max(0, int(round(truth["population"] + pop_noise)))
            tti_obs = max(0, int(round(truth["tti"] + tti_noise)))

            obs.severity = round(sev_obs, 2)
            obs.population_at_risk = int(pop_obs)
            obs.time_to_impact = int(tti_obs)

            obs.severity_uncertainty = _clamp(abs(sev_obs - truth["severity"]) / 4.0 + self._profile.obs_noise * 0.6)
            obs.population_uncertainty = _clamp(abs(pop_obs - truth["population"]) / max(truth["population"], 1.0) + self._profile.obs_noise * 0.4)
            obs.tti_uncertainty = _clamp(abs(tti_obs - truth["tti"]) / max(self._episode_total_steps, 1) + self._profile.obs_noise * 0.5)

            estimated_priority = (obs.severity * max(obs.population_at_risk, 1)) / max(obs.time_to_impact, 1)
            obs.priority_score = round(float(estimated_priority), 3)
            obs.risk_level = "high" if estimated_priority >= 520 else "medium" if estimated_priority >= 140 else "low"

            if obs.risk_level == "high" and obs.time_to_impact <= 3:
                obs.recommended_action_hint = "rescue_or_allocate"
            elif obs.tti_uncertainty > 0.2 or obs.population_uncertainty > 0.2:
                obs.recommended_action_hint = "predict"
            else:
                obs.recommended_action_hint = "coordinate"

            visible_threats.append(obs)

        resources = [r.model_copy(deep=True) for r in self._resources]
        zones = [z.model_copy(deep=True) for z in self._affected_zones]

        return CrisisObservation(
            threats=visible_threats,
            resources=resources,
            affected_zones=zones,
            time_remaining=max(0, self._episode_total_steps - self._step_count),
            current_step=self._step_count,
            alerts=alerts,
            episode_id=self._episode_id,
            resource_budget_remaining=self._resource_budget_remaining,
            resource_budget_total=self._resource_budget_total,
            recent_actions=list(self._recent_actions[-5:]),
            valid_actions=self.valid_actions(),
        )

    # ─────────────────────────────────────────
    # INTERNAL UTILS
    # ─────────────────────────────────────────

    def _adaptive_task_weights(
        self,
        performance: Dict[str, float],
        alpha: float = ADAPTIVE_ALPHA,
        floor: float = ADAPTIVE_WEIGHT_FLOOR,
    ) -> Dict[str, float]:
        _ = alpha
        _ = floor
        raw = {}
        for task_name in CORE_TASKS:
            perf = _clamp(float(performance.get(task_name, 0.0)))
            raw[task_name] = float(TASK_IMPORTANCE.get(task_name, 1.0)) * (1.0 - perf)
        total = sum(raw.values())
        if total <= 0:
            return {k: 1.0 / len(CORE_TASKS) for k in CORE_TASKS}
        return {k: raw[k] / total for k in CORE_TASKS}

    def _is_high_priority_rescue_target(self, zone_id: int) -> bool:
        threat = self._get_threat(zone_id)
        if threat is None:
            return False
        truth = self._true_state.get(threat.threat_id, {})
        severity_high = float(truth.get("severity", 0.0)) >= 7.0
        population_high = float(truth.get("population", 0.0)) >= 700.0
        low_initial_tti = float(truth.get("initial_tti", 999.0)) <= 5.0
        return bool(severity_high or population_high or low_initial_tti)

    def _action_balance_kl(self, valid_actions_payload: Optional[Dict[str, object]] = None) -> float:
        if valid_actions_payload is None:
            valid_actions_payload = self.valid_actions()

        mask = valid_actions_payload.get("action_mask", [])
        valid_core: List[str] = []
        for action_name in CORE_ACTIONS:
            idx = ACTION_TYPE_ORDER.index(action_name)
            if idx < len(mask) and int(mask[idx]) == 1:
                valid_core.append(action_name)

        if len(valid_core) <= 1:
            return 0.0

        counts = np.asarray([float(self._action_counts[a]) + 1e-6 for a in valid_core], dtype=np.float64)
        p = counts / np.sum(counts)

        q_raw = np.asarray(
            [float(self._baseline_action_dist.get(a, 0.0)) + 1e-6 for a in valid_core],
            dtype=np.float64,
        )
        q = q_raw / np.sum(q_raw)
        kl = float(np.sum(p * np.log(p / q)))
        return max(0.0, kl)

    def _action_coverage_bonus(self, action_name: str, prev_last_action_step: int) -> float:
        if action_name not in CORE_ACTIONS:
            return 0.0
        valid_payload = self.valid_actions()
        mask = valid_payload.get("action_mask", [])
        idx = ACTION_TYPE_ORDER.index(action_name)
        if idx >= len(mask) or int(mask[idx]) == 0:
            return 0.0
        steps_since = self._step_count - prev_last_action_step
        if steps_since >= COVERAGE_WINDOW:
            return SOFT_COVERAGE_BONUS
        return 0.0

    def _terminal_high_risk_boost(self) -> float:
        high_risk_ids: List[int] = []
        for threat in self._threats:
            truth = self._true_state.get(threat.threat_id, {})
            severity_high = float(truth.get("severity", 0.0)) >= 7.0
            population_high = float(truth.get("population", 0.0)) >= 700.0
            low_initial_tti = float(truth.get("initial_tti", 999.0)) <= 5.0
            if severity_high or population_high or low_initial_tti:
                high_risk_ids.append(threat.threat_id)

        if not high_risk_ids:
            return 0.0

        handled = 0
        for tid in high_risk_ids:
            threat = self._get_threat(tid)
            if threat is None:
                continue
            if threat.status in {ThreatStatus.CONTAINED, ThreatStatus.RESOLVED}:
                handled += 1
                continue
            if threat.status == ThreatStatus.IMPACTED:
                zone = self._get_zone(tid)
                if zone is not None and zone.total_victims > 0:
                    rescued_ratio = zone.rescued / max(zone.total_victims, 1)
                    if rescued_ratio >= 0.7:
                        handled += 1

        handled_ratio = handled / max(len(high_risk_ids), 1)
        if handled_ratio >= 0.8:
            return TERMINAL_HIGH_RISK_BOOST
        return 0.0

    @staticmethod
    def _normalize_step_reward(raw_reward: float) -> float:
        # Keep upper bound for score stability while preserving negative gradient signal.
        return float(min(1.0, raw_reward))

    def _generate_threats(self, threat_count: int) -> List[ThreatInfo]:
        chosen = self._rng.sample(THREAT_TEMPLATES, k=threat_count)
        threats: List[ThreatInfo] = []
        for i, (threat_type, zone, location, base_pop) in enumerate(chosen, start=1):
            severity = round(self._rng.uniform(4.0, 9.6), 2)
            pop_jitter = self._rng.randint(-int(base_pop * 0.22), int(base_pop * 0.22))
            population = max(15, base_pop + pop_jitter)
            tti = self._rng.randint(4, 16)
            threats.append(
                ThreatInfo(
                    threat_id=i,
                    threat_type=threat_type,
                    status=ThreatStatus.ACTIVE,
                    severity=severity,
                    population_at_risk=population,
                    time_to_impact=tti,
                    zone=zone,
                    location_name=location,
                )
            )
        return threats

    def _generate_resources(self) -> List[ResourceInfo]:
        resources: List[ResourceInfo] = []
        for i, (rtype, zone, base_eff) in enumerate(RESOURCE_TEMPLATES, start=1):
            jitter = self._rng.uniform(-0.06, 0.06)
            resources.append(
                ResourceInfo(
                    resource_id=i,
                    resource_type=rtype,
                    is_available=True,
                    effectiveness=round(_clamp(base_eff + jitter), 3),
                    location_zone=zone,
                    cooldown_steps=0,
                )
            )
        return resources

    def _spawn_secondary_threat(self, parent: ThreatInfo) -> ThreatInfo:
        next_id = max(t.threat_id for t in self._threats) + 1
        cascade_type = self._rng.choice([ThreatType.FIRE, ThreatType.EXPLOSION, ThreatType.DRONE_THREAT])

        parent_truth = self._true_state[parent.threat_id]
        severity = round(_clamp(parent_truth["severity"] * self._rng.uniform(0.48, 0.72), 2.5, 8.8), 2)
        population = max(20, int(parent_truth["population"] * self._rng.uniform(0.25, 0.45)))
        tti = self._rng.randint(3, 9)

        new = ThreatInfo(
            threat_id=next_id,
            threat_type=cascade_type,
            status=ThreatStatus.ACTIVE,
            severity=severity,
            population_at_risk=population,
            time_to_impact=tti,
            zone=parent.zone,
            location_name=f"{parent.location_name} (secondary)",
        )

        self._threats.append(new)
        self._true_state[next_id] = {
            "severity": float(severity),
            "population": float(population),
            "tti": float(tti),
            "initial_tti": float(tti),
        }
        self._delay_buffer[next_id] = 0
        self._threat_mitigation[next_id] = 0.0
        self._total_population += population

        return new

    def _stable_noise(self, threat_id: int, channel: int, scale: float) -> float:
        mix = (
            int(self._seed or 0) * 1_000_003
            + self._step_count * 97_403
            + threat_id * 4_759
            + channel * 389
        )
        rng = random.Random(mix)
        return rng.gauss(0.0, scale)

    def _record_action_memory(self, action: CrisisAction) -> None:
        action_type = action.action_type
        name = action_type.value if hasattr(action_type, "value") else str(action_type)
        self._recent_actions.append(name)
        if len(self._recent_actions) > 8:
            self._recent_actions = self._recent_actions[-8:]
        if name in self._action_counts:
            self._action_counts[name] += 1
            self._last_action_step[name] = self._step_count

    def _spend_budget(self, amount: int) -> None:
        amount = max(0, int(amount))
        self._resource_budget_remaining = max(0, self._resource_budget_remaining - amount)
        self._resource_spent += amount

    def _is_critical(self, threat: ThreatInfo) -> bool:
        return self._true_priority(threat) >= 520 or self._true_state[threat.threat_id]["tti"] <= 2

    def _true_priority(self, threat: ThreatInfo) -> float:
        truth = self._true_state[threat.threat_id]
        return (truth["severity"] * max(truth["population"], 1.0)) / max(truth["tti"], 1.0)

    def _highest_risk_active_threat(self) -> Optional[ThreatInfo]:
        active = [t for t in self._threats if t.status == ThreatStatus.ACTIVE]
        if not active:
            return None
        return max(active, key=self._true_priority)

    def _get_threat(self, threat_id: int) -> Optional[ThreatInfo]:
        for threat in self._threats:
            if threat.threat_id == threat_id:
                return threat
        return None

    def _get_resource(self, resource_id: int) -> Optional[ResourceInfo]:
        for resource in self._resources:
            if resource.resource_id == resource_id:
                return resource
        return None

    def _get_zone(self, zone_id: int) -> Optional[AffectedZoneInfo]:
        for zone in self._affected_zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    @staticmethod
    def _rank_correlation_score(agent_order: List[int], ideal_order: List[int]) -> float:
        if not ideal_order:
            return 0.0

        n = len(ideal_order)
        ideal_rank = {tid: idx for idx, tid in enumerate(ideal_order)}
        filtered = [tid for tid in agent_order if tid in ideal_rank]
        if not filtered:
            return 0.0

        weighted_error = 0.0
        total_weight = 0.0
        for tid, agent_idx in zip(filtered, range(len(filtered))):
            ideal_idx = ideal_rank[tid]
            weight = 1.0 / (ideal_idx + 1)
            err = abs(agent_idx - ideal_idx) / max(n - 1, 1)
            weighted_error += weight * err
            total_weight += weight

        if total_weight <= 0:
            return 0.0
        return round(_clamp(1.0 - (weighted_error / total_weight)), 4)
