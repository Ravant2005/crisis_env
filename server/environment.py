"""
server/environment.py — Core simulation engine for the AI Crisis Response & Rescue Coordination Environment.

Design goals:
- True decision-making MDP/POMDP: any valid action at any step.
- Non-deterministic dynamics with seed-reproducibility.
- Explicit trade-offs: limited global resource budget and time pressure.
- Action masking support for efficient RL training.
- Reward aligned with final score to avoid objective mismatch.

BUG FIXES applied in this version:
  1. Removed forced-classify override that hijacked every non-classify action.
  2. Fixed allocation budget guard: was hardcoded <=3, now dynamic.
  3. Fixed _grader_rescue: no longer gives free 0.40 baseline before any rescue.
  4. Fixed step reward: now driven by real task-score deltas, not near-zero constants.
  5. Fixed coordination: no longer requires 2 *currently active* threats.
  6. Fixed resource distance field: guarded against missing attribute.
  7. Step print now feeds reward.py-compatible C/P/A/Co/R breakdown.
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

GLOBAL_RESOURCE_BUDGET = 5

ZONE_RESOURCE_AFFINITY: Dict[ZoneType, List[ResourceType]] = {
    ZoneType.MILITARY: [ResourceType.MILITARY_UNIT, ResourceType.MEDICAL_TEAM],
    ZoneType.MARITIME: [ResourceType.COAST_GUARD, ResourceType.RESCUE_DRONE],
    ZoneType.URBAN:    [ResourceType.SWAT_TEAM, ResourceType.FIRE_BRIGADE, ResourceType.EVACUATION_BUS],
    ZoneType.RURAL:    [ResourceType.FIRE_BRIGADE, ResourceType.MEDICAL_TEAM, ResourceType.RESCUE_DRONE],
}

THREAT_TEMPLATES: List[Tuple[ThreatType, ZoneType, str, int]] = [
    (ThreatType.AIRSTRIKE,    ZoneType.MILITARY, "Military Base Alpha",        70),
    (ThreatType.SHIP_ATTACK,  ZoneType.MARITIME, "Naval Port Sector 7",        240),
    (ThreatType.DRONE_THREAT, ZoneType.URBAN,    "Downtown Business District", 1200),
    (ThreatType.EXPLOSION,    ZoneType.URBAN,    "Central Train Station",      900),
    (ThreatType.FLOOD,        ZoneType.RURAL,    "River Valley District",      450),
    (ThreatType.FIRE,         ZoneType.URBAN,    "Industrial Complex East",    360),
]

RESOURCE_TEMPLATES: List[Tuple[ResourceType, ZoneType, float]] = [
    (ResourceType.MILITARY_UNIT,   ZoneType.MILITARY, 0.90),
    (ResourceType.COAST_GUARD,     ZoneType.MARITIME, 0.88),
    (ResourceType.SWAT_TEAM,       ZoneType.URBAN,    0.85),
    (ResourceType.FIRE_BRIGADE,    ZoneType.URBAN,    0.82),
    (ResourceType.MEDICAL_TEAM,    ZoneType.MILITARY, 0.78),
    (ResourceType.RESCUE_DRONE,    ZoneType.MARITIME, 0.72),
    (ResourceType.EVACUATION_BUS,  ZoneType.URBAN,    0.66),
    (ResourceType.MEDICAL_TEAM,    ZoneType.RURAL,    0.76),
]

ACTION_TYPE_ORDER: List[str] = [
    "classify", "predict", "allocate", "coordinate", "rescue", "skip", "delay",
]

STRATEGIES: List[str] = ["rescue_first", "predict_first", "balanced"]
CORE_ACTIONS: List[str] = ["classify", "predict", "allocate", "coordinate", "rescue"]
CORE_TASKS:   List[str] = ["classification", "prediction", "allocation", "coordination", "rescue"]

TASK_IMPORTANCE: Dict[str, float] = {
    "classification": 1.0,
    "prediction":     1.0,
    "allocation":     1.6,
    "coordination":   1.5,
    "rescue":         1.8,
}

# Reward shaping constants
LAMBDA_TIME            = 0.006
LAMBDA_RESOURCE_WASTE  = 1.0
LAMBDA_PRIORITY_ERROR  = 1.0
COVERAGE_WINDOW        = 10
SOFT_COVERAGE_BONUS    = 0.02
TERMINAL_HIGH_RISK_BOOST = 0.10

DEFAULT_BASELINE_ACTION_DIST: Dict[str, float] = {
    "classify":   0.22,
    "predict":    0.21,
    "allocate":   0.22,
    "coordinate": 0.18,
    "rescue":     0.17,
}

# Minimum step reward — keeps gradient signal alive but doesn't mask task progress
MIN_STEP_REWARD = 0.02


@dataclass
class DifficultyProfile:
    name:             str
    threats:          int
    obs_noise:        float
    escalation_prob:  float
    spread_prob:      float
    budget:           int
    episode_steps:    int


DIFFICULTY_PROFILES: Dict[str, DifficultyProfile] = {
    "easy":   DifficultyProfile("easy",   threats=2, obs_noise=0.08, escalation_prob=0.08, spread_prob=0.05, budget=10, episode_steps=22),
    "medium": DifficultyProfile("medium", threats=3, obs_noise=0.16, escalation_prob=0.15, spread_prob=0.12, budget=9,  episode_steps=26),
    "hard":   DifficultyProfile("hard",   threats=4, obs_noise=0.24, escalation_prob=0.23, spread_prob=0.20, budget=8,  episode_steps=30),
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
      - step(action)              -> StepResult
      - state()                   -> CrisisState
      - task_scores()             -> Dict[str, float]
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed if seed is not None else 42
        self._rng  = random.Random(self._seed)
        random.seed(self._seed)
        import numpy as np
        np.random.seed(self._seed)

        self._difficulty          = "medium"
        self._profile             = DIFFICULTY_PROFILES[self._difficulty]
        self._episode_total_steps = self._profile.episode_steps

        self._episode_id  = ""
        self._step_count  = 0
        self._done        = False

        self._threats:        List[ThreatInfo]        = []
        self._resources:      List[ResourceInfo]       = []
        self._affected_zones: List[AffectedZoneInfo]   = []

        self._true_state:        Dict[int, Dict[str, float]] = {}
        self._delay_buffer:      Dict[int, int]              = {}
        self._threat_mitigation: Dict[int, float]            = {}

        self._resource_budget_total     = GLOBAL_RESOURCE_BUDGET
        self._resource_budget_remaining = GLOBAL_RESOURCE_BUDGET
        self._resource_spent            = 0
        self._wasted_resource_events    = 0

        self._total_population    = 0
        self._casualties          = 0
        self._casualties_prevented = 0
        self._cumulative_reward   = 0.0

        self._classify_scores: Dict[int, float] = {}
        self._predict_scores:  Dict[int, float] = {}
        self._alloc_scores:    List[float]       = []
        self._coord_scores:    List[float]       = []
        self._rescue_total_victims = 0
        self._rescue_saved         = 0
        self._rescue_steps:    List[int]         = []

        self._wrong_priority_events  = 0
        self._critical_ignored_steps = 0

        self._coordinated = False
        self._classified: set = set()
        self._predicted:  set = set()
        self._allocated:  set = set()

        self._previous_final_score  = 0.0
        self._final_score_history:  List[float] = []

        self._recent_actions:    List[str]       = []
        self._action_counts:     Dict[str, int]  = {a: 0 for a in CORE_ACTIONS}
        self._last_action_step:  Dict[str, int]  = {a: -999 for a in CORE_ACTIONS}
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
        random.seed(self._seed)
        import numpy as np
        np.random.seed(self._seed)

        self._difficulty          = difficulty if difficulty in DIFFICULTY_PROFILES else "medium"
        self._profile             = DIFFICULTY_PROFILES[self._difficulty]
        self._episode_total_steps = self._profile.episode_steps

        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._done       = False

        self._threats        = self._generate_threats(self._profile.threats)
        self._resources      = self._generate_resources()
        self._affected_zones = []

        self._true_state = {
            t.threat_id: {
                "severity":    float(t.severity),
                "population":  float(t.population_at_risk),
                "tti":         float(t.time_to_impact),
                "initial_tti": float(t.time_to_impact),
            }
            for t in self._threats
        }
        self._delay_buffer      = {t.threat_id: 0   for t in self._threats}
        self._threat_mitigation = {t.threat_id: 0.0 for t in self._threats}

        self._resource_budget_total     = self._profile.budget
        self._resource_budget_remaining = self._profile.budget
        self._resource_spent            = 0
        self._wasted_resource_events    = 0

        self._total_population    = sum(int(v["population"]) for v in self._true_state.values())
        self._casualties          = 0
        self._casualties_prevented = 0
        self._cumulative_reward   = 0.0

        self._classify_scores = {}
        self._predict_scores  = {}
        self._alloc_scores    = []
        self._coord_scores    = []
        self._rescue_total_victims = 0
        self._rescue_saved         = 0
        self._rescue_streak        = 0
        self._rescue_steps         = []
        self._wrong_priority_events  = 0
        self._critical_ignored_steps = 0

        self._recent_actions  = []
        self._action_counts   = {a: 0 for a in CORE_ACTIONS}
        self._last_action_step = {a: -999 for a in CORE_ACTIONS}
        self._coordinated      = False
        self._classified       = set()
        self._predicted        = set()
        self._allocated        = set()
        self._previous_final_score = 0.0
        self._final_score_history  = []

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

        # Snapshot task scores BEFORE action so we can compute deltas
        prev_task_scores = self.task_scores()

        action_name = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )
        prev_last_action_step = int(self._last_action_step.get(action_name, -999))

        # Keep tracked sets current
        all_threat_ids = {t.threat_id for t in self._threats}
        self._classified = self._classified & all_threat_ids
        self._predicted  = self._predicted  & all_threat_ids
        self._allocated  = self._allocated  & all_threat_ids

        # ── PROCESS ACTION ────────────────────────────────────────────────
        # BUG FIX 1: Removed forced-classify override.
        # The original code hijacked every non-classify step to force a classify
        # action, which prevented the agent from ever learning predict/allocate/
        # coordinate/rescue. Deleted entirely.
        # ─────────────────────────────────────────────────────────────────

        alerts: List[str] = []
        info:   Dict[str, float] = {}

        action_bonus, action_alerts, action_info = self._process_action(action)
        alerts.extend(action_alerts)
        info.update(action_info)

        resource_waste_penalty = float(action_info.get("resource_waste_penalty", 0.0))
        wrong_priority_penalty = float(action_info.get("wrong_priority_penalty", 0.0))

        lifecycle_alerts, lifecycle_info = self._advance_dynamics()
        alerts.extend(lifecycle_alerts)

        resource_waste_penalty += float(lifecycle_info.get("resource_waste_penalty", 0.0))
        wrong_priority_penalty += float(lifecycle_info.get("wrong_priority_penalty", 0.0))

        self._step_count += 1

        no_active_threats = not any(t.status == ThreatStatus.ACTIVE for t in self._threats)
        no_active_zones   = not any(z.is_active for z in self._affected_zones)
        time_up           = self._step_count >= self._episode_total_steps

        if (no_active_threats and no_active_zones) or time_up:
            self._done = True

        # ── TASK-SCORE DELTA REWARD ───────────────────────────────────────
        # BUG FIX 4: Step reward is now driven by real per-task-score deltas.
        # Previously the delta was near-zero every step because the graders
        # barely changed, locking reward at the 0.02 floor.
        current_task_scores = self.task_scores()

        d_classify = max(0.0, current_task_scores["classification"] - prev_task_scores["classification"])
        d_predict  = max(0.0, current_task_scores["prediction"]     - prev_task_scores["prediction"])
        d_alloc    = max(0.0, current_task_scores["allocation"]      - prev_task_scores["allocation"])
        d_coord    = max(0.0, current_task_scores["coordination"]    - prev_task_scores["coordination"])
        d_rescue   = max(0.0, current_task_scores["rescue"]          - prev_task_scores["rescue"])

        # Weighted task-progress component (weights sum to 1.0)
        task_progress = (
            0.20 * d_classify +
            0.20 * d_predict  +
            0.20 * d_alloc    +
            0.15 * d_coord    +
            0.25 * d_rescue
        )

        # Action-specific bonus from the handler (small, 0.01-0.06)
        handler_bonus = max(0.0, float(action_bonus))

        # Budget efficiency bonus: reward spending resources (not hoarding)
        budget_eff = _clamp(self._resource_spent / max(self._resource_budget_total, 1))

        # Time pressure penalty: mild discount for late actions
        time_fraction = self._step_count / max(self._episode_total_steps, 1)

        # Penalty for invalid actions
        invalid_action_penalty = 0.0
        valid_info = self.valid_actions()
        amask = valid_info.get("action_mask", [])
        try:
            aidx = ACTION_TYPE_ORDER.index(action_name)
            if aidx < len(amask) and int(amask[aidx]) == 0:
                invalid_action_penalty = 0.15
        except ValueError:
            pass

        # Waste penalty
        waste_frac = _clamp(self._wasted_resource_events / max(self._episode_total_steps, 1))

        # Base Reward Component
        reward = (
            2.5 * task_progress
            + handler_bonus
            + 0.15 * budget_eff
            - 0.20 * time_fraction
            - 0.35 * invalid_action_penalty
            - 0.25 * waste_frac
        )

        # FIX 2: Skip / Delay Suppression & Progress Check
        if action_name == "skip":
            reward *= 0.2
        elif action_name == "delay":
            reward *= 0.4
        
        # FIX 3: No Progress Penalty
        if task_progress < 0.01:
            reward *= 0.3


        # FIX 4: Task Completion Boost
        if d_classify > 0: reward += 0.1
        if d_predict > 0:  reward += 0.1
        if d_alloc > 0:    reward += 0.1

        # FIX 1: Strong rescue reward shaping
        if action_name == "rescue":
            total_people = max(self._rescue_total_victims, 1)
            improvement = self._rescue_saved / total_people
            reward += 0.2 * improvement
            if improvement > 0.8:
                reward += 0.1
            if d_rescue > 0:
                reward += 0.2

        # FIX 5: Pipeline Bonus
        # Intelligent decision pipeline: classify -> predict -> allocate
        classify_done = len(self._classified) >= len(self._threats) and len(self._threats) > 0
        predict_done  = len(self._predicted) >= len(self._threats) and len(self._threats) > 0
        allocate_done = len(self._allocated) >= len(self._threats) and len(self._threats) > 0
        
        if classify_done and predict_done:
            reward += 0.2
        if predict_done and allocate_done:
            reward += 0.2

        # 🔥 FIX 7: Force Early Termination (Cleaner Episodes)
        rescue_done = len(self._affected_zones) > 0 and all(not z.is_active for z in self._affected_zones)
        total_rescued_ratio = self._rescue_saved / max(self._rescue_total_victims, 1) if self._rescue_total_victims > 0 else 0
        if rescue_done or self._resource_budget_remaining <= 0 or total_rescued_ratio > 0.85:
            self._done = True

        # 🔥 FIX 6: No progress penalty (Stricter penalization)
        if abs(task_progress) < 0.01:
            reward -= 0.05

        # Terminal bonus for completing the episode cleanly
        if self._done:
            # FIX 2: Terminal bonus
            final_score = (
                0.20 * current_task_scores["classification"] +
                0.20 * current_task_scores["prediction"] +
                0.20 * current_task_scores["allocation"] +
                0.15 * current_task_scores["coordination"] +
                0.25 * current_task_scores["rescue"]
            )
            reward += 0.1 * final_score

        # FIX 1: Remove floor and use float np.clip
        step_reward = float(np.clip(reward, -0.15, 0.85))

        self._cumulative_reward += step_reward

        # Clamp to [MIN_STEP_REWARD, 1.0] — never negative (keeps value network stable)
        # step_reward = max(MIN_STEP_REWARD, min(1.0, reward))
        # REMOVED RAW PRINT PER FIX 10: print(f"[STEP {self._step_count}] ...")

        info.update({
            "step_task_progress":    round(task_progress,          6),
            "budget_efficiency":     round(budget_eff,             6),
            "time_efficiency":       round(1.0 - time_fraction,    6),
            "invalid_action_penalty":round(invalid_action_penalty, 6),
            "resource_waste_penalty":round(waste_frac,             6),
            "d_classify":            round(d_classify,             6),
            "d_predict":             round(d_predict,              6),
            "d_alloc":               round(d_alloc,                6),
            "d_coord":               round(d_coord,                6),
            "d_rescue":              round(d_rescue,               6),
        })

        obs = self._build_observation(alerts)
        return StepResult(observation=obs, reward=round(step_reward, 6), done=self._done, info=info)

    def state(self) -> CrisisState:
        c_score  = self._grader_classification()
        p_score  = self._grader_prediction()
        a_score  = self._grader_allocation()
        co_score = self._grader_coordination()
        r_score  = self._grader_rescue()
        final    = self._compute_final_score(cached=(c_score, p_score, a_score, co_score, r_score))

        total_victims = max(self._rescue_total_victims, 1)
        rescue_rate   = _clamp(self._rescue_saved / total_victims)

        resolved = sum(
            1 for t in self._threats
            if t.status in {ThreatStatus.CONTAINED, ThreatStatus.RESOLVED}
        )

        return CrisisState(
            step_count               = self._step_count,
            total_steps              = self._episode_total_steps,
            episode_id               = self._episode_id,
            difficulty               = self._difficulty,
            classification_score     = c_score,
            prediction_score         = p_score,
            allocation_score         = a_score,
            coordination_score       = co_score,
            rescue_score             = r_score,
            final_score              = final,
            resolved_threats         = resolved,
            total_threats            = len(self._threats),
            casualties               = self._casualties,
            casualties_prevented     = self._casualties_prevented,
            total_population_at_risk = self._total_population,
            rescue_success_rate      = round(rescue_rate, 4),
            resource_budget_remaining = self._resource_budget_remaining,
            resource_budget_total    = self._resource_budget_total,
            cumulative_reward        = round(self._cumulative_reward, 6),
            done                     = self._done,
            task_scores={
                "classification": round(c_score,  3),
                "prediction":     round(p_score,  3),
                "allocation":     round(a_score,  3),
                "coordination":   round(co_score, 3),
                "rescue":         round(r_score,  3),
            },
            efficiency={
                "budget": round(self._resource_spent / max(self._resource_budget_total, 1), 3),
                "time":   round(1.0 - (self._step_count / max(self._episode_total_steps, 1)), 3),
            },
            progress={
                "classified": len(self._classified),
                "predicted":  len(self._predicted),
                "allocated":  len(self._allocated),
                "rescued":    self._rescue_saved,
            },
        )

    def task_scores(self) -> Dict[str, float]:
        return {
            "classification": self._grader_classification(),
            "prediction":     self._grader_prediction(),
            "allocation":     self._grader_allocation(),
            "coordination":   self._grader_coordination(),
            "rescue":         self._grader_rescue(),
        }

    def set_baseline_action_distribution(self, action_dist: Dict[str, float]) -> None:
        cleaned = {a: max(float(action_dist.get(a, 0.0)), 0.0) for a in CORE_ACTIONS}
        total = sum(cleaned.values())
        if total <= 0:
            self._baseline_action_dist = dict(DEFAULT_BASELINE_ACTION_DIST)
            return
        self._baseline_action_dist = {a: cleaned[a] / total for a in CORE_ACTIONS}

    def valid_actions(self) -> Dict[str, object]:
        active_threats     = sorted([t for t in self._threats      if t.status == ThreatStatus.ACTIVE], key=lambda t: t.threat_id)
        active_zones       = sorted([z for z in self._affected_zones if z.is_active],                   key=lambda z: z.zone_id)
        available_resources= sorted([r for r in self._resources     if r.is_available],                 key=lambda r: r.resource_id)

        # BUG FIX 5: coordinate allowed if we have ever seen ≥2 threats, even if only 1 active now
        all_seen = [t for t in self._threats]
        can_coordinate = len(all_seen) >= 2

        # 🔥 FIX 1: ELIMINATE SKIP SPAM (HARD CONTROL)
        has_valid_actions = int(bool(active_threats)) + int(bool(active_threats and available_resources and self._resource_budget_remaining > 0)) + int(can_coordinate) + int(bool(active_zones and self._resource_budget_remaining > 0)) > 0
        
        action_mask = {
            "classify":   int(bool(active_threats)),
            "predict":    int(bool(active_threats)),
            "allocate":   int(bool(active_threats and available_resources and self._resource_budget_remaining > 0)),
            "coordinate": int(can_coordinate),
            "rescue":     int(bool(active_zones and self._resource_budget_remaining > 0)),
            # 🚫 ONLY allow skip if NOTHING else is valid
            "skip":       0 if has_valid_actions else 1,
            "delay":      int(bool(active_threats)),
        }

        return {
            "action_types":    list(ACTION_TYPE_ORDER),
            "action_mask":     [action_mask[a] for a in ACTION_TYPE_ORDER],
            "threat_ids":      [t.threat_id   for t in active_threats],
            "resource_ids":    [r.resource_id for r in available_resources],
            "zone_ids":        [z.zone_id     for z in active_zones],
            "max_rescue_units": min(MAX_RESCUE_UNITS, max(0, self._resource_budget_remaining)),
            "strategy_options": list(STRATEGIES),
        }

    # ─────────────────────────────────────────
    # ACTION HANDLERS
    # ─────────────────────────────────────────

    def _process_action(self, action: CrisisAction) -> Tuple[float, List[str], Dict[str, float]]:
        self._record_action_memory(action)

        if action.action_type == ActionType.CLASSIFY   and action.classification:
            return self._handle_classify(action.classification)
        if action.action_type == ActionType.PREDICT    and action.prediction:
            return self._handle_predict(action.prediction)
        if action.action_type == ActionType.ALLOCATE   and action.allocation:
            return self._handle_allocate(action.allocation)
        if action.action_type == ActionType.COORDINATE and action.coordination:
            return self._handle_coordinate(action.coordination)
        if action.action_type == ActionType.RESCUE     and action.rescue:
            return self._handle_rescue(action.rescue)
        if action.action_type == ActionType.DELAY      and action.delay:
            return self._handle_delay(action.delay)
        if action.action_type == ActionType.SKIP:
            return 0.0, ["[SKIP] Holding action this step."], {}
        if action.action_type == ActionType.EVACUATE   and action.evacuate:
            return self._handle_evacuate(action.evacuate)

        return -0.02, ["[WARN] Invalid or incomplete action payload."], {"resource_waste_penalty": 0.01}

    def _handle_classify(self, payload: ClassificationPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat = self._get_threat(payload.threat_id)
        if threat is None or threat.status != ThreatStatus.ACTIVE:
            return -0.02, ["[WARN] CLASSIFY target invalid or not active."], {"resource_waste_penalty": 0.01}

        truth     = self._true_state[threat.threat_id]
        type_correct = payload.predicted_type == threat.threat_type
        sev_error    = abs(float(payload.predicted_severity) - float(truth["severity"]))
        tolerance    = 1.0 + self._profile.obs_noise * 2.5

        if type_correct and sev_error <= tolerance:
            score = 1.0
            label = "CORRECT"
        elif type_correct:
            score = 0.55
            label = "PARTIAL"
        else:
            score = 0.0
            label = "WRONG"

        self._classify_scores[threat.threat_id] = max(
            self._classify_scores.get(threat.threat_id, 0.0), score
        )
        self._classified.add(threat.threat_id)
        threat.predicted_severity = float(payload.predicted_severity)

        bonus = score * 0.04
        return bonus, [f"[CLASSIFY] Threat {threat.threat_id} -> {label} (sev_err={sev_error:.2f})"], {}

    def _handle_predict(self, payload: PredictionPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat = self._get_threat(payload.threat_id)
        if threat is None or threat.status != ThreatStatus.ACTIVE:
            return -0.02, ["[WARN] PREDICT target invalid or not active."], {"resource_waste_penalty": 0.01}

        truth     = self._true_state[threat.threat_id]
        tti_error = abs(float(payload.predicted_tti) - float(truth["tti"]))
        pop_error = abs(float(payload.predicted_pop) - float(truth["population"])) / max(float(truth["population"]), 1.0)

        norm_tti = _clamp(tti_error / max(self._episode_total_steps, 1))
        error    = 0.5 * norm_tti + 0.5 * _clamp(pop_error)
        score    = _clamp(1.0 - error)

        self._predict_scores[threat.threat_id] = max(
            self._predict_scores.get(threat.threat_id, 0.0), score
        )
        self._predicted.add(threat.threat_id)
        # Defensive: ensure numeric conversion
        try:
            threat.predicted_tti = int(payload.predicted_tti) if isinstance(payload.predicted_tti, (int, float)) else 0
        except (ValueError, TypeError):
            threat.predicted_tti = 0
        try:
            threat.predicted_pop = int(payload.predicted_pop) if isinstance(payload.predicted_pop, (int, float)) else 0
        except (ValueError, TypeError):
            threat.predicted_pop = 0

        bonus = score * 0.04
        return bonus, [f"[PREDICT] Threat {threat.threat_id} -> quality={score:.3f}"], {}

    def _handle_allocate(self, payload: AllocationPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat   = self._get_threat(payload.threat_id)
        resource = self._get_resource(payload.resource_id)

        if threat is None or threat.status != ThreatStatus.ACTIVE or resource is None or not resource.is_available:
            return -0.03, ["[WARN] ALLOCATE invalid threat/resource."], {"resource_waste_penalty": 0.02}

        if self._resource_budget_remaining <= 0:
            self._wasted_resource_events += 1
            return -0.04, ["[WARN] ALLOCATE blocked: resource budget exhausted."], {"resource_waste_penalty": 0.04}

        # BUG FIX 2: Dynamic rescue reserve instead of hardcoded <=3.
        # Reserve budget = number of still-active zones (each needs ≥1 unit).
        # This prevents over-blocking while still protecting rescue capacity.
        active_zones_count = sum(1 for z in self._affected_zones if z.is_active)
        reserve_needed = max(1, active_zones_count)
        if self._resource_budget_remaining <= reserve_needed and active_zones_count > 0:
            self._wasted_resource_events += 1
            return -0.02, [f"[WARN] ALLOCATE blocked: {reserve_needed} units reserved for {active_zones_count} active rescue zones."], {"resource_waste_penalty": 0.02}

        self._spend_budget(1)
        resource.is_available = False
        resource.assigned_to  = threat.threat_id
        resource.cooldown_steps = self._rng.randint(1, 3)
        threat.assigned_resource = resource.resource_id

        affinity     = ZONE_RESOURCE_AFFINITY.get(threat.zone, [])
        match_bonus  = 0.18 if resource.resource_type in affinity else -0.08

        effectiveness_noise = self._rng.uniform(-0.12, 0.12)
        effective_power = _clamp(resource.effectiveness + match_bonus + effectiveness_noise)
        self._threat_mitigation[threat.threat_id] = _clamp(
            self._threat_mitigation.get(threat.threat_id, 0.0) + effective_power * 0.55,
            0.0, 0.95,
        )

        alloc_score = _clamp(resource.effectiveness + (0.2 if resource.resource_type in affinity else 0.0))
        self._alloc_scores.append(alloc_score)
        self._allocated.add(threat.threat_id)

        wrong_priority_penalty = 0.0
        highest = self._highest_risk_active_threat()
        if highest is not None and highest.threat_id != threat.threat_id:
            chosen_risk  = self._true_priority(threat)
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
        if self._coordinated:
            return -0.05, ["Redundant coordination"], {}
            
        # BUG FIX 5: Coordinate works on all ever-seen threats, not just currently active ones.
        # Original required ≥2 *currently* active threats, which often blocked coordination
        # in the mid/late episode when threats had already impacted.
        all_tracked = [t for t in self._threats]
        if len(all_tracked) < 2:
            return -0.01, ["[WARN] COORDINATE ignored: fewer than 2 threats tracked."], {"resource_waste_penalty": 0.005}

        active = sorted(
            [t for t in self._threats if t.status == ThreatStatus.ACTIVE],
            key=lambda t: t.threat_id,
        )
        # Build ordering scope: prefer active, fall back to all tracked
        scope = active if len(active) >= 2 else all_tracked
        scope_ids = {t.threat_id for t in scope}

        order = [tid for tid in payload.priority_order if tid in scope_ids]
        if not order:
            # Accept any order that mentions known threat IDs
            order = [tid for tid in payload.priority_order if tid in {t.threat_id for t in all_tracked}]
        if not order:
            return -0.02, ["[WARN] COORDINATE ignored: no valid threat IDs in payload."], {"resource_waste_penalty": 0.01}

        ideal = [t.threat_id for t in sorted(scope, key=self._true_priority, reverse=True)]
        score = self._rank_correlation_score(order, ideal)
        self._coord_scores.append(score)

        for rank, tid in enumerate(order):
            t = self._get_threat(tid)
            if t is not None:
                t.priority_rank = rank + 1

        wrong_priority_penalty = 0.0
        if ideal and order[0] != ideal[0]:
            wrong_priority_penalty = 0.015
            self._wrong_priority_events += 1

        bonus = score * 0.05
        self._coordinated = True
        return bonus, [f"[COORDINATE] priority={order} | score={score:.3f}"], {
            "wrong_priority_penalty": wrong_priority_penalty
        }

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
            ZoneType.URBAN:    1.00,
            ZoneType.MARITIME: 0.85,
            ZoneType.MILITARY: 0.90,
            ZoneType.RURAL:    0.78,
        }.get(zone.zone_type, 1.0)

        rescue_effectiveness = _clamp(
            self._rng.uniform(0.76, 1.08) - self._profile.obs_noise * 0.2, 0.45, 1.15
        )
        saved = min(remaining, int(spendable * 14 * zone_multiplier * rescue_effectiveness))
        zone.rescued       += saved
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

        speed_factor = _clamp(1.0 - (self._step_count / max(self._episode_total_steps, 1)))
        
        improvement = saved / max(zone.total_victims, 1)
        bonus        = (improvement * 0.6) + speed_factor * 0.01
        
        # 🔥 FIX 5: RESCUE MOMENTUM BONUS
        if improvement > 0:
            self._rescue_streak += 1
            bonus += min(0.3, 0.05 * self._rescue_streak)
        else:
            self._rescue_streak = 0
            
        high_priority = self._is_high_priority_rescue_target(zone.zone_id)

        return bonus, [
            f"[RESCUE] Zone {zone.zone_id}: saved={saved}, remaining={zone.total_victims - zone.rescued}"
        ], {
            "resource_waste_penalty":  resource_waste_penalty,
            "high_priority_rescue":    1.0 if high_priority else 0.0,
        }

    def _handle_delay(self, payload: DelayPayload) -> Tuple[float, List[str], Dict[str, float]]:
        threat = self._get_threat(payload.threat_id)
        if threat is None or threat.status != ThreatStatus.ACTIVE:
            return -0.02, ["[WARN] DELAY target invalid or not active."], {"resource_waste_penalty": 0.01}

        requested    = int(payload.delay_steps)
        success_prob = _clamp(
            0.62
            - self._profile.obs_noise * 0.7
            + self._threat_mitigation.get(threat.threat_id, 0.0) * 0.2,
            0.2, 0.85,
        )

        if self._rng.random() < success_prob:
            self._delay_buffer[threat.threat_id] = min(
                3, self._delay_buffer.get(threat.threat_id, 0) + requested
            )
            bonus = 0.012 * requested
            return bonus, [f"[DELAY] Threat {threat.threat_id} delayed by {requested} step(s)."], {}

        truth = self._true_state[threat.threat_id]
        truth["severity"] = min(10.0, truth["severity"] + self._rng.uniform(0.1, 0.5))
        truth["population"] *= self._rng.uniform(1.01, 1.06)
        return -0.01, [f"[DELAY] Threat {threat.threat_id} delay failed; escalation observed."], {
            "wrong_priority_penalty": 0.008
        }

    def _handle_evacuate(self, payload: EvacuationPayload) -> Tuple[float, List[str], Dict[str, float]]:
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
            "delay_penalty":          0.0,
        }

        for resource in self._resources:
            if resource.cooldown_steps > 0:
                resource.cooldown_steps -= 1
                if resource.cooldown_steps == 0:
                    resource.is_available = True
                    resource.assigned_to  = None

        active_threats = [t for t in self._threats if t.status == ThreatStatus.ACTIVE]

        for threat in active_threats:
            tid        = threat.threat_id
            truth      = self._true_state[tid]
            mitigation = self._threat_mitigation.get(tid, 0.0)

            escalation_prob = _clamp(self._profile.escalation_prob * (1.1 - mitigation), 0.02, 0.45)
            if self._rng.random() < escalation_prob:
                truth["severity"]   = min(10.0, truth["severity"] + self._rng.uniform(0.2, 0.8))
                truth["population"] *= self._rng.uniform(1.03, 1.10)
                alerts.append(f"[ESCALATION] Threat {tid} intensified.")

            if (
                len(self._threats) < MAX_THREATS_VISIBLE
                and truth["severity"] >= 7.0
                and self._rng.random() < self._profile.spread_prob * (1.0 - mitigation)
            ):
                new = self._spawn_secondary_threat(parent=threat)
                alerts.append(f"[SPREAD] Secondary threat {new.threat_id} emerged near {threat.location_name}.")

            if self._delay_buffer.get(tid, 0) > 0:
                self._delay_buffer[tid] -= 1
            else:
                truth["tti"] = max(0.0, truth["tti"] - 1.0)

            self._threat_mitigation[tid] = _clamp(mitigation * 0.92, 0.0, 0.95)

            if truth["tti"] > 0 and self._threat_mitigation[tid] > 0.72 and self._rng.random() < 0.35:
                threat.status = ThreatStatus.CONTAINED
                prevented = int(truth["population"] * 0.55)
                self._casualties_prevented += prevented
                threat.casualties_prevented += prevented
                alerts.append(f"[CONTAINED] Threat {tid} neutralized before impact.")
                continue

            if truth["tti"] <= 0.0:
                casualties, prevented = self._compute_impact(threat)
                threat.status              = ThreatStatus.IMPACTED
                threat.casualties          = casualties
                threat.casualties_prevented = prevented

                self._casualties           += casualties
                self._casualties_prevented += prevented

                zone = AffectedZoneInfo(
                    zone_id       = threat.threat_id,
                    zone_type     = threat.zone,
                    location_name = threat.location_name,
                    total_victims = casualties,
                    is_active     = casualties > 0,
                )
                self._affected_zones.append(zone)
                self._rescue_total_victims += max(0, casualties)

                alerts.append(
                    f"[IMPACT] Threat {tid} impacted at {threat.location_name}. "
                    f"victims={casualties}, prevented={prevented}"
                )

        for zone in self._affected_zones:
            if zone.is_active:
                continue
            threat = self._get_threat(zone.zone_id)
            if threat and threat.status == ThreatStatus.IMPACTED and zone.rescued >= zone.total_victims:
                threat.status = ThreatStatus.RESOLVED
                alerts.append(f"[RESOLVED] Threat {threat.threat_id} resolved after rescue.")

        critical = [
            t for t in self._threats
            if t.status == ThreatStatus.ACTIVE and self._is_critical(t)
        ]
        unhandled_critical = [
            t for t in critical
            if t.assigned_resource is None and self._delay_buffer.get(t.threat_id, 0) == 0
        ]
        if unhandled_critical:
            penalty = 0.006 * len(unhandled_critical)
            info["delay_penalty"] += penalty
            self._critical_ignored_steps += len(unhandled_critical)

        return alerts, info

    def _compute_impact(self, threat: ThreatInfo) -> Tuple[int, int]:
        truth      = self._true_state[threat.threat_id]
        severity   = _clamp(truth["severity"] / 10.0)
        population = max(0, int(truth["population"]))

        mitigation              = self._threat_mitigation.get(threat.threat_id, 0.0)
        prediction_support      = self._predict_scores.get(threat.threat_id,   0.0) * 0.08
        classification_support  = self._classify_scores.get(threat.threat_id,  0.0) * 0.05
        total_mitigation        = _clamp(mitigation + prediction_support + classification_support, 0.0, 0.92)

        base_rate    = _clamp(0.08 + severity * 0.72, 0.05, 0.95)
        chaos_factor = self._rng.uniform(0.85, 1.15)
        casualty_rate = _clamp(base_rate * chaos_factor * (1.0 - total_mitigation), 0.01, 0.98)

        casualties = int(population * casualty_rate)
        prevented  = max(0, population - casualties)
        return casualties, prevented

    # ─────────────────────────────────────────
    # GRADERS
    # ─────────────────────────────────────────

    def _grader_classification(self) -> float:
        relevant = [
            t for t in self._threats
            if t.status in {ThreatStatus.ACTIVE, ThreatStatus.IMPACTED,
                            ThreatStatus.RESOLVED, ThreatStatus.CONTAINED}
        ]
        if not relevant:
            return 0.0

        correct_count  = 0
        severity_errors = []
        for t in relevant:
            score = self._classify_scores.get(t.threat_id, 0.0)
            if score >= 0.55:
                correct_count += 1
            if score > 0:
                truth = self._true_state[t.threat_id]
                severity_errors.append(
                    abs(float(t.predicted_severity) - float(truth["severity"])) / 10.0
                )

        correct_ratio    = correct_count / len(relevant)
        severity_accuracy = (1.0 - _clamp(_mean(severity_errors))) if severity_errors else 0.0

        final_score = 0.70 * correct_ratio + 0.30 * severity_accuracy
        return round(_clamp(final_score), 4)

    def _grader_prediction(self) -> float:
        relevant = [
            t for t in self._threats
            if t.status in {ThreatStatus.ACTIVE, ThreatStatus.IMPACTED,
                            ThreatStatus.RESOLVED, ThreatStatus.CONTAINED}
        ]
        if not relevant:
            return 0.0

        tti_errors = []
        pop_errors = []
        for t in relevant:
            score = self._predict_scores.get(t.threat_id, 0.0)
            if score > 0:
                truth   = self._true_state[t.threat_id]
                tti_err = _clamp(abs(int(t.predicted_tti) - int(truth["tti"])) / max(self._episode_total_steps, 1))
                pop_err = _clamp(abs(int(t.predicted_pop) - int(truth["population"])) / max(int(truth["population"]), 1))
                tti_errors.append(tti_err)
                pop_errors.append(pop_err)

        if not tti_errors:
            return 0.0

        score = 1.0 - (0.5 * _clamp(_mean(tti_errors)) + 0.5 * _clamp(_mean(pop_errors)))
        return round(_clamp(score), 4)

    def _grader_allocation(self) -> float:
        if not self._alloc_scores:
            return 0.0

        effectiveness = _mean(self._alloc_scores)

        affinity_hits  = 0
        affinity_total = 0
        for t in self._threats:
            if t.threat_id in self._allocated and t.assigned_resource:
                r = self._get_resource(t.assigned_resource)
                if r:
                    affinity = ZONE_RESOURCE_AFFINITY.get(t.zone, [])
                    affinity_hits  += int(r.resource_type in affinity)
                    affinity_total += 1
        zone_affinity  = affinity_hits / max(affinity_total, 1)

        spent_ratio      = _clamp(self._resource_spent / max(self._resource_budget_total, 1), 0.0, 1.5)
        budget_efficiency = 1.0 - abs(spent_ratio - 1.0) if spent_ratio <= 1.0 else 0.0

        waste_ratio = self._wasted_resource_events / max(self._episode_total_steps, 1)
        waste_penalty = 0.02 * min(self._wasted_resource_events, 8) / 8.0

        score = (
            0.45 * effectiveness +
            0.30 * zone_affinity +
            0.15 * budget_efficiency -
            waste_penalty
        )
        return round(_clamp(score), 4)

    def _grader_coordination(self) -> float:
        if not self._coord_scores:
            return 0.0

        base          = _clamp(_mean(self._coord_scores))
        wrong_penalty = 0.04 * min(self._wrong_priority_events, 6) / 6.0
        score         = base - wrong_penalty
        return round(_clamp(score), 4)

    def _grader_rescue(self) -> float:
        # BUG FIX 3: Removed the free 0.40 baseline that was handed out before any
        # rescue action was taken. That caused R: to show 0.40 from step 1.
        # Now we only give credit once actual rescue or prevention has occurred.

        if self._rescue_total_victims <= 0:
            # No zones hit yet — score entirely on casualties *prevented* so far
            if self._total_population <= 0:
                return 0.0
            preventive = _clamp(self._casualties_prevented / max(self._total_population, 1))
            # Only credit if meaningful prevention has happened (>0)
            return round(_clamp(preventive * 0.50), 4)

        saved_ratio = _clamp(self._rescue_saved / max(self._rescue_total_victims, 1))

        if self._rescue_steps:
            avg_step    = _mean([float(s) for s in self._rescue_steps])
            speed_score = _clamp(1.0 - avg_step / max(self._episode_total_steps, 1))
        else:
            speed_score = 0.0

        deployed = sum(z.rescue_units_deployed for z in self._affected_zones)
        # Relax resource efficiency constraint since budget is tight
        resource_efficiency = _clamp(self._rescue_saved / max(1, deployed * 8))

        score = 0.65 * saved_ratio + 0.20 * speed_score + 0.15 * resource_efficiency
        urgency = 1.0 + (self._step_count / max(self._episode_total_steps, 1))
        score *= min(1.2, urgency)
        return round(_clamp(score), 4)

    def _compute_final_score(
        self,
        cached: Optional[Tuple[float, float, float, float, float]] = None,
    ) -> float:
        if cached is None:
            c  = self._grader_classification()
            p  = self._grader_prediction()
            a  = self._grader_allocation()
            co = self._grader_coordination()
            r  = self._grader_rescue()
        else:
            c, p, a, co, r = cached

        final = (
            0.20 * c  +
            0.20 * p  +
            0.20 * a  +
            0.15 * co +
            0.25 * r
        )
        return round(_clamp(final), 4)

    # ─────────────────────────────────────────
    # OBSERVATION BUILDER
    # ─────────────────────────────────────────

    def _build_observation(self, alerts: List[str]) -> CrisisObservation:
        visible_threats: List[ThreatInfo] = []
        for threat in self._threats:
            if not isinstance(threat.threat_id, int):
                print(f"[ENV WARN] Invalid threat_id type: {type(threat.threat_id)} value={threat.threat_id}")
                continue
            truth = self._true_state[threat.threat_id]
            obs   = threat.model_copy(deep=True)

            sev_noise = self._stable_noise(threat.threat_id, 11, self._profile.obs_noise * 2.2)
            pop_noise = self._stable_noise(threat.threat_id, 17, max(20.0, truth["population"] * self._profile.obs_noise * 0.18))
            tti_noise = self._stable_noise(threat.threat_id, 23, max(0.4,  self._profile.obs_noise * 2.5))

            sev_obs = _clamp(truth["severity"]   + sev_noise, 0.0, 10.0)
            pop_obs = max(0,  int(round(truth["population"] + pop_noise)))
            tti_obs = max(0,  int(round(truth["tti"]        + tti_noise)))

            obs.severity          = round(sev_obs, 2)
            obs.population_at_risk = int(pop_obs)
            obs.time_to_impact    = int(tti_obs)

            obs.severity_uncertainty    = _clamp(abs(sev_obs - truth["severity"]) / 4.0 + self._profile.obs_noise * 0.6)
            obs.population_uncertainty  = _clamp(abs(pop_obs - truth["population"]) / max(truth["population"], 1.0) + self._profile.obs_noise * 0.4)
            obs.tti_uncertainty         = _clamp(abs(tti_obs - truth["tti"]) / max(self._episode_total_steps, 1) + self._profile.obs_noise * 0.5)

            est_priority        = (obs.severity * max(obs.population_at_risk, 1)) / max(obs.time_to_impact, 1)
            obs.priority_score  = round(float(est_priority), 3)
            obs.risk_level      = "high" if est_priority >= 520 else "medium" if est_priority >= 140 else "low"

            if obs.risk_level == "high" and obs.time_to_impact <= 3:
                obs.recommended_action_hint = "rescue_or_allocate"
            elif obs.tti_uncertainty > 0.2 or obs.population_uncertainty > 0.2:
                obs.recommended_action_hint = "predict"
            else:
                obs.recommended_action_hint = "coordinate"

            visible_threats.append(obs)

        resources = [r.model_copy(deep=True) for r in self._resources]
        zones     = [z.model_copy(deep=True) for z in self._affected_zones]

        return CrisisObservation(
            threats                  = visible_threats,
            resources                = resources,
            affected_zones           = zones,
            time_remaining           = max(0, self._episode_total_steps - self._step_count),
            current_step             = self._step_count,
            alerts                   = alerts,
            episode_id               = self._episode_id,
            resource_budget_remaining = self._resource_budget_remaining,
            resource_budget_total    = self._resource_budget_total,
            recent_actions           = list(self._recent_actions[-5:]),
            valid_actions            = self.valid_actions(),
        )

    # ─────────────────────────────────────────
    # INTERNAL UTILS
    # ─────────────────────────────────────────

    def _is_high_priority_rescue_target(self, zone_id: int) -> bool:
        threat = self._get_threat(zone_id)
        if threat is None:
            return False
        truth         = self._true_state.get(threat.threat_id, {})
        severity_high = float(truth.get("severity",    0.0)) >= 7.0
        pop_high      = float(truth.get("population",  0.0)) >= 700.0
        low_tti       = float(truth.get("initial_tti", 999.0)) <= 5.0
        return bool(severity_high or pop_high or low_tti)

    def _action_balance_kl(self, valid_actions_payload: Optional[Dict[str, object]] = None) -> float:
        if valid_actions_payload is None:
            valid_actions_payload = self.valid_actions()

        mask       = valid_actions_payload.get("action_mask", [])
        valid_core = [
            a for a in CORE_ACTIONS
            if ACTION_TYPE_ORDER.index(a) < len(mask)
               and int(mask[ACTION_TYPE_ORDER.index(a)]) == 1
        ]
        if len(valid_core) <= 1:
            return 0.0

        counts = np.asarray([float(self._action_counts[a]) + 1e-6 for a in valid_core], dtype=np.float64)
        p      = counts / np.sum(counts)
        q_raw  = np.asarray([float(self._baseline_action_dist.get(a, 0.0)) + 1e-6 for a in valid_core], dtype=np.float64)
        q      = q_raw / np.sum(q_raw)
        return max(0.0, float(np.sum(p * np.log(p / q))))

    def _action_coverage_bonus(self, action_name: str, prev_last_action_step: int) -> float:
        if action_name not in CORE_ACTIONS:
            return 0.0
        valid_payload = self.valid_actions()
        mask = valid_payload.get("action_mask", [])
        idx  = ACTION_TYPE_ORDER.index(action_name)
        if idx >= len(mask) or int(mask[idx]) == 0:
            return 0.0
        if self._step_count - prev_last_action_step >= COVERAGE_WINDOW:
            return SOFT_COVERAGE_BONUS
        return 0.0

    def _terminal_high_risk_boost(self) -> float:
        high_risk_ids = [
            t.threat_id for t in self._threats
            if (
                float(self._true_state.get(t.threat_id, {}).get("severity",    0.0)) >= 7.0
                or float(self._true_state.get(t.threat_id, {}).get("population", 0.0)) >= 700.0
                or float(self._true_state.get(t.threat_id, {}).get("initial_tti", 999.0)) <= 5.0
            )
        ]
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
                    if zone.rescued / max(zone.total_victims, 1) >= 0.7:
                        handled += 1

        if handled / max(len(high_risk_ids), 1) >= 0.8:
            return TERMINAL_HIGH_RISK_BOOST
        return 0.0

    def _generate_threats(self, threat_count: int) -> List[ThreatInfo]:
        chosen   = self._rng.sample(THREAT_TEMPLATES, k=threat_count)
        threats: List[ThreatInfo] = []
        for i, (threat_type, zone, location, base_pop) in enumerate(chosen, start=1):
            severity  = round(self._rng.uniform(4.0, 9.6), 2)
            pop_jitter = self._rng.randint(-int(base_pop * 0.22), int(base_pop * 0.22))
            population = max(15, base_pop + pop_jitter)
            tti        = self._rng.randint(4, 16)
            threats.append(ThreatInfo(
                threat_id        = i,
                threat_type      = threat_type,
                status           = ThreatStatus.ACTIVE,
                severity         = severity,
                population_at_risk = population,
                time_to_impact   = tti,
                zone             = zone,
                location_name    = location,
            ))
        return threats

    def _generate_resources(self) -> List[ResourceInfo]:
        resources: List[ResourceInfo] = []
        for i, (rtype, zone, base_eff) in enumerate(RESOURCE_TEMPLATES, start=1):
            jitter = self._rng.uniform(-0.06, 0.06)
            resources.append(ResourceInfo(
                resource_id    = i,
                resource_type  = rtype,
                is_available   = True,
                effectiveness  = round(_clamp(base_eff + jitter), 3),
                location_zone  = zone,
                cooldown_steps = 0,
                # BUG FIX 6: distance attribute populated so train.py reward_kwargs builder
                # doesn't crash on missing field.
                distance       = round(self._rng.uniform(1.0, 50.0), 1),
            ))
        return resources

    def _spawn_secondary_threat(self, parent: ThreatInfo) -> ThreatInfo:
        next_id      = max(t.threat_id for t in self._threats) + 1
        cascade_type = self._rng.choice([ThreatType.FIRE, ThreatType.EXPLOSION, ThreatType.DRONE_THREAT])

        parent_truth = self._true_state[parent.threat_id]
        severity     = round(_clamp(parent_truth["severity"] * self._rng.uniform(0.48, 0.72), 2.5, 8.8), 2)
        population   = max(20, int(parent_truth["population"] * self._rng.uniform(0.25, 0.45)))
        tti          = self._rng.randint(3, 9)

        new = ThreatInfo(
            threat_id         = next_id,
            threat_type       = cascade_type,
            status            = ThreatStatus.ACTIVE,
            severity          = severity,
            population_at_risk = population,
            time_to_impact    = tti,
            zone              = parent.zone,
            location_name     = f"{parent.location_name} (secondary)",
        )

        self._threats.append(new)
        self._true_state[next_id] = {
            "severity":    float(severity),
            "population":  float(population),
            "tti":         float(tti),
            "initial_tti": float(tti),
        }
        self._delay_buffer[next_id]      = 0
        self._threat_mitigation[next_id] = 0.0
        self._total_population          += population
        return new

    def _stable_noise(self, threat_id: int, channel: int, scale: float) -> float:
        mix = (
            int(self._seed or 0) * 1_000_003
            + self._step_count   * 97_403
            + threat_id          * 4_759
            + channel            * 389
        )
        return random.Random(mix).gauss(0.0, scale)

    def _record_action_memory(self, action: CrisisAction) -> None:
        name = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        self._recent_actions.append(name)
        if len(self._recent_actions) > 8:
            self._recent_actions = self._recent_actions[-8:]
        if name in self._action_counts:
            self._action_counts[name]    += 1
            self._last_action_step[name]  = self._step_count

    def _spend_budget(self, amount: int) -> None:
        amount = max(0, int(amount))
        self._resource_budget_remaining = max(0, self._resource_budget_remaining - amount)
        self._resource_spent           += amount

    def _is_critical(self, threat: ThreatInfo) -> bool:
        return self._true_priority(threat) >= 520 or self._true_state[threat.threat_id]["tti"] <= 2

    def _true_priority(self, threat: ThreatInfo) -> float:
        truth = self._true_state[threat.threat_id]
        return (truth["severity"] * max(truth["population"], 1.0)) / max(truth["tti"], 1.0)

    def _highest_risk_active_threat(self) -> Optional[ThreatInfo]:
        active = [t for t in self._threats if t.status == ThreatStatus.ACTIVE]
        return max(active, key=self._true_priority) if active else None

    def _get_threat(self, threat_id: int) -> Optional[ThreatInfo]:
        for t in self._threats:
            if t.threat_id == threat_id:
                return t
        return None

    def _get_resource(self, resource_id: int) -> Optional[ResourceInfo]:
        for r in self._resources:
            if r.resource_id == resource_id:
                return r
        return None

    def _get_zone(self, zone_id: int) -> Optional[AffectedZoneInfo]:
        for z in self._affected_zones:
            if z.zone_id == zone_id:
                return z
        return None

    @staticmethod
    def _rank_correlation_score(agent_order: List[int], ideal_order: List[int]) -> float:
        if not ideal_order:
            return 0.0
        n          = len(ideal_order)
        ideal_rank = {tid: idx for idx, tid in enumerate(ideal_order)}
        filtered   = [tid for tid in agent_order if tid in ideal_rank]
        if not filtered:
            return 0.0

        weighted_error = 0.0
        total_weight   = 0.0
        for tid, agent_idx in zip(filtered, range(len(filtered))):
            ideal_idx    = ideal_rank[tid]
            weight       = 1.0 / (ideal_idx + 1)
            err          = abs(agent_idx - ideal_idx) / max(n - 1, 1)
            weighted_error += weight * err
            total_weight   += weight

        if total_weight <= 0:
            return 0.0
        return round(_clamp(1.0 - weighted_error / total_weight), 4)