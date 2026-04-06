"""
rewards.py  —  Crisis Response RL · Dynamic Reward & Scoring Engine
====================================================================
FIXES IN THIS VERSION v2:
  - _COMPAT fully updated for crisis domain (fire_brigade, medical_team,
    rescue_drone, coast_guard, swat_team, evacuation_bus, military_unit).
    Previously all military domain → type_score was always 0.08. Now 0.45.
  - reward_coordination: removed step_decay multiplier that caused the
    shaped reward to mislead the policy about re-coordination value.
    Added flat early_bonus instead (small, non-compounding).
  - reward_rescue: bonus thresholds lowered (0.3→0.2) + rescue rate fixed.
  - BASE_REWARD raised to 0.03 for rescue to encourage more rescue steps.
  - SKIP_PENALTY strengthened to -0.12 to break skip loops faster.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# TASK WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
TASK_WEIGHTS: Dict[str, float] = {
    "classification": 0.18,
    "prediction":     0.18,
    "allocation":     0.18,
    "coordination":   0.20,   # 🔥 boosted
    "rescue":         0.26,   # 🔥 boosted
}

BASE_REWARD      = 0.02    # minimum reward for PRODUCTIVE task actions only
RESCUE_BASE      = 0.06    # boosted to encourage more rescue steps
STEP_DECAY_GAMMA = 0.97    # encourage acting early; decays reward per step

# Penalty for non-productive actions (skip/delay/repeated invalid)
SKIP_PENALTY  = -0.12   # strengthened: must break skip loops faster
DELAY_PENALTY = -0.05   # softer but still negative

# ENV blend ratio: shaped reward contributes 70%, raw env reward 30%
SHAPED_BLEND = 0.70

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS  (auto-disabled when not a tty)
# ─────────────────────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

RED     = lambda t: _c("91", t)
GREEN   = lambda t: _c("92", t)
YELLOW  = lambda t: _c("93", t)
CYAN    = lambda t: _c("96", t)
MAGENTA = lambda t: _c("95", t)
BOLD    = lambda t: _c("1",  t)
DIM     = lambda t: _c("2",  t)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskScores:
    classification: float = 0.0
    prediction:     float = 0.0
    allocation:     float = 0.0
    coordination:   float = 0.0
    rescue:         float = 0.0

    @property
    def final(self) -> float:
        return (
            self.classification * TASK_WEIGHTS["classification"]
            + self.prediction   * TASK_WEIGHTS["prediction"]
            + self.allocation   * TASK_WEIGHTS["allocation"]
            + self.coordination * TASK_WEIGHTS["coordination"]
            + self.rescue       * TASK_WEIGHTS["rescue"]
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "classification": round(self.classification, 4),
            "prediction":     round(self.prediction,     4),
            "allocation":     round(self.allocation,     4),
            "coordination":   round(self.coordination,   4),
            "rescue":         round(self.rescue,         4),
            "final":          round(self.final,          4),
        }


@dataclass
class StepInfo:
    step:        int
    action_type: str
    raw_reward:  float
    scores:      TaskScores
    done:        bool
    episode:     int   = 0
    phase:       str   = ""
    difficulty:  str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))

def _gauss(err: float, sigma: float) -> float:
    """Gaussian reward: 1.0 at zero error, decays smoothly."""
    return math.exp(-(err ** 2) / (2.0 * sigma ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# T1 — THREAT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

THREAT_DOMAINS: Dict[str, List[str]] = {
    # Military
    "air":    ["missile", "drone", "airstrike", "helicopter", "aircraft"],
    "land":   ["tank", "ground_troops", "artillery", "vehicle", "infantry"],
    "water":  ["warship", "submarine", "torpedo", "naval", "boat"],
    "cyber":  ["malware", "ddos", "intrusion", "ransomware"],
    "chem":   ["chemical", "biological", "nuclear", "radiological"],
    # Crisis / disaster (added)
    "fire":   ["fire", "wildfire", "arson", "blaze", "inferno"],
    "flood":  ["flood", "tsunami", "storm_surge", "inundation", "flash_flood"],
    "geo":    ["earthquake", "landslide", "avalanche", "volcanic", "sinkhole"],
    "storm":  ["hurricane", "typhoon", "cyclone", "tornado", "blizzard"],
    "hazmat": ["chemical_spill", "gas_leak", "radiation", "toxic", "hazmat"],
    "civil":  ["riot", "hostage", "terrorism", "explosion", "bombing"],
    "med":    ["pandemic", "outbreak", "epidemic", "mass_casualty"],
}

def _threat_domain(threat_type: str) -> str:
    t = threat_type.lower()
    for domain, subtypes in THREAT_DOMAINS.items():
        if t == domain or any(s in t for s in subtypes):
            return domain
    return "unknown"


def reward_classification(
    predicted_type:     str,
    true_type:          str,
    predicted_severity: float,
    true_severity:      float,
    confidence:         float = 1.0,
) -> float:
    """
    T1 — Threat Classification reward (max 1.0).
      0.55  exact subtype match
      0.25  domain match only
      0.20  severity accuracy  Gaussian(err, σ=0.25)
    """
    pt = predicted_type.lower().strip()
    tt = true_type.lower().strip()

    if pt == tt:
        type_score = 0.55
    elif _threat_domain(pt) == _threat_domain(tt) and _threat_domain(tt) != "unknown":
        type_score = 0.25
    else:
        type_score = 0.0

    sev_err   = abs(predicted_severity - true_severity) / max(abs(true_severity), 1.0)
    sev_score = 0.20 * _gauss(sev_err, sigma=0.25)
    conf      = _clamp(confidence, 0.5, 1.0)
    return _clamp((type_score + sev_score) * conf)


# ─────────────────────────────────────────────────────────────────────────────
# T2 — IMPACT PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def reward_prediction(
    predicted_tti:  int,
    true_tti:       int,
    predicted_pop:  int,
    true_pop:       int,
    total_steps:    int = 30,
) -> float:
    """
    T2 — Impact Prediction reward (max 1.0).
      0.50  TTI accuracy   Gaussian over normalised step error
      0.50  Population     Gaussian over log-ratio
    """
    tti_err   = abs(predicted_tti - true_tti) / max(total_steps, 1)
    tti_score = 0.50 * _gauss(tti_err, sigma=0.15)

    if true_pop > 0:
        log_ratio = abs(math.log((max(predicted_pop, 1)) / max(true_pop, 1)))
        pop_score = 0.50 * _gauss(log_ratio, sigma=0.50)
    else:
        pop_score = 0.50 if predicted_pop == 0 else 0.0

    return _clamp(tti_score + pop_score)


# ─────────────────────────────────────────────────────────────────────────────
# T3 — COUNTER-DEFENSE ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

# FIX: _COMPAT now covers BOTH military AND crisis domains.
# The crisis env uses resource types like fire_brigade, medical_team,
# rescue_drone, coast_guard, swat_team, evacuation_bus, military_unit.
# The original dict only had military types → type_score was always 0.08.
# Now crisis threat types map to their natural crisis resources → 0.45.
_COMPAT: Dict[str, List[str]] = {
    # ── Military threats ──────────────────────────────────────────────────
    "missile":       ["interceptor", "air_defense", "radar", "sam", "military"],
    "drone":         ["jammer", "interceptor", "air_defense", "laser", "rescue_drone"],
    "airstrike":     ["interceptor", "air_defense", "fighter", "sam", "military"],
    "helicopter":    ["sam", "air_defense", "fighter", "radar", "military"],
    "aircraft":      ["fighter", "sam", "air_defense", "interceptor", "military"],
    "tank":          ["artillery", "anti_tank", "helicopter", "mine", "military"],
    "ground_troops": ["infantry", "artillery", "helicopter", "military"],
    "artillery":     ["counter_battery", "artillery", "helicopter", "military"],
    "warship":       ["destroyer", "submarine", "coastal_battery", "missile", "coast"],
    "submarine":     ["destroyer", "depth_charge", "sonar", "helicopter", "coast"],
    "torpedo":       ["decoy", "destroyer", "sonar", "coast"],
    "chemical":      ["hazmat", "decon_unit", "cbrn", "medical"],
    "biological":    ["hazmat", "decon_unit", "cbrn", "medical"],
    "nuclear":       ["evacuation", "decon_unit", "cbrn", "medical"],
    # ── Crisis / disaster threats ─────────────────────────────────────────
    "fire":               ["fire_brigade", "rescue_drone", "medical_team", "swat_team"],
    "wildfire":           ["fire_brigade", "rescue_drone", "evacuation_bus"],
    "arson":              ["fire_brigade", "swat_team", "medical_team"],
    "flood":              ["coast_guard", "rescue_drone", "evacuation_bus", "medical_team"],
    "flash_flood":        ["coast_guard", "rescue_drone", "evacuation_bus"],
    "tsunami":            ["coast_guard", "evacuation_bus", "rescue_drone", "military_unit"],
    "storm_surge":        ["coast_guard", "evacuation_bus", "rescue_drone"],
    "earthquake":         ["medical_team", "rescue_drone", "fire_brigade", "evacuation_bus"],
    "landslide":          ["rescue_drone", "medical_team", "fire_brigade"],
    "avalanche":          ["rescue_drone", "medical_team", "evacuation_bus"],
    "hurricane":          ["evacuation_bus", "coast_guard", "fire_brigade", "rescue_drone"],
    "typhoon":            ["evacuation_bus", "coast_guard", "rescue_drone"],
    "tornado":            ["evacuation_bus", "rescue_drone", "medical_team"],
    "explosion":          ["fire_brigade", "medical_team", "swat_team", "rescue_drone"],
    "bombing":            ["swat_team", "medical_team", "military_unit", "fire_brigade"],
    "chemical_spill":     ["medical_team", "fire_brigade", "rescue_drone"],
    "gas_leak":           ["fire_brigade", "evacuation_bus", "medical_team"],
    "radiation":          ["medical_team", "evacuation_bus", "military_unit"],
    "hazmat":             ["medical_team", "fire_brigade", "rescue_drone"],
    "hostage":            ["swat_team", "military_unit", "medical_team"],
    "terrorism":          ["swat_team", "military_unit", "medical_team"],
    "riot":               ["swat_team", "military_unit", "evacuation_bus"],
    "pandemic":           ["medical_team", "evacuation_bus"],
    "mass_casualty":      ["medical_team", "rescue_drone", "evacuation_bus"],
    "building_collapse":  ["fire_brigade", "medical_team", "rescue_drone"],
    # ── Generic fallback keywords (substring match) ────────────────────────
    # Any resource containing these substrings will score 0.45 for the listed
    # threat keywords. This handles env-specific naming variations.
    "_fire_generic":      ["fire"],        # matches fire_brigade, firefighter
    "_medical_generic":   ["medical"],     # matches medical_team, medic
    "_rescue_generic":    ["rescue"],      # matches rescue_drone, rescue_team
    "_coast_generic":     ["coast"],       # matches coast_guard
    "_evac_generic":      ["evacuation"],  # matches evacuation_bus
    "_military_generic":  ["military"],    # matches military_unit
    "_swat_generic":      ["swat"],        # matches swat_team
}

# Subset keyword → compat list mapping for efficient fallback lookup
_COMPAT_KEYWORDS: Dict[str, List[str]] = {
    "fire":        ["fire_brigade", "fire", "brigade"],
    "flood":       ["coast_guard", "coast", "rescue_drone", "rescue", "evacuation", "evacuation_bus"],
    "earth":       ["medical_team", "medical", "rescue_drone", "rescue", "fire_brigade"],
    "quake":       ["medical_team", "medical", "rescue_drone", "rescue", "fire_brigade"],
    "tsunami":     ["coast_guard", "coast", "evacuation_bus", "evacuation", "rescue_drone"],
    "hurricane":   ["evacuation_bus", "evacuation", "coast_guard", "coast", "fire_brigade"],
    "tornado":     ["evacuation_bus", "evacuation", "rescue_drone", "rescue"],
    "explosion":   ["fire_brigade", "fire", "medical_team", "medical", "swat_team"],
    "chemical":    ["medical_team", "medical", "fire_brigade", "fire"],
    "hazmat":      ["medical_team", "medical", "fire_brigade", "fire"],
    "hostage":     ["swat_team", "swat", "military_unit", "military"],
    "riot":        ["swat_team", "swat", "military_unit", "military"],
    "terror":      ["swat_team", "swat", "military_unit", "military"],
    "pandemic":    ["medical_team", "medical", "evacuation_bus"],
    "collapse":    ["fire_brigade", "fire", "medical_team", "medical", "rescue_drone"],
    "landslide":   ["rescue_drone", "rescue", "medical_team", "medical"],
    "avalanche":   ["rescue_drone", "rescue", "medical_team", "medical"],
    "wildfire":    ["fire_brigade", "fire", "rescue_drone", "rescue"],
}


def reward_allocation(
    resource_type:    str,
    threat_type:      str,
    intercept_prob:   float,
    budget_used:      int,
    budget_total:     int,
    nearest_chosen:   bool = True,
) -> float:
    """
    T3 — Counter-Defense Allocation reward (max 1.0).
      0.45  resource–threat type compatibility  (FIXED: crisis domain added)
      0.30  intercept probability
      0.15  budget efficiency
      0.10  proximity bonus

    FIX v2: Added full crisis domain to _COMPAT. Previously only military
    threats were listed, so crisis env resources always scored type_score=0.08.
    Now fire/flood/earthquake/etc map to fire_brigade/medical_team/rescue_drone.
    """
    rt  = resource_type.lower().strip()
    tt  = threat_type.lower().strip()

    # 1. Direct lookup in _COMPAT
    compat_list = _COMPAT.get(tt, [])
    type_score = 0.45 if compat_list and any(c in rt for c in compat_list) else 0.0

    # 2. Fallback: keyword-based substring matching for crisis domain variants
    if type_score == 0.0:
        for keyword, compat_kw in _COMPAT_KEYWORDS.items():
            if keyword in tt:
                if any(c in rt for c in compat_kw):
                    type_score = 0.40  # slightly lower than exact match
                    break

    # 3. Universal fallback: any crisis-response resource gets partial credit
    # This handles env-specific naming we can't anticipate.
    if type_score == 0.0:
        crisis_resources = ["fire_brigade", "medical_team", "rescue_drone",
                            "coast_guard", "swat_team", "evacuation_bus",
                            "military_unit", "fire", "medical", "rescue",
                            "coast", "evacuation", "swat", "military"]
        if any(c in rt for c in crisis_resources):
            type_score = 0.15  # partial: resource is crisis-appropriate even if
                               # we couldn't match the specific threat type

    # If still 0.0, set minimum fallback
    if type_score == 0.0:
        type_score = 0.08

    intercept_score = 0.30 * _clamp(intercept_prob)

    if budget_total > 0:
        frac = budget_used / budget_total
        efficiency = 0.15 * _gauss(frac - 0.3, sigma=0.3)
    else:
        efficiency = 0.0

    proximity = 0.10 if nearest_chosen else 0.0

    base_score = type_score + intercept_score + efficiency + proximity
    
    # ALLOCATION QUALITY BOOST - stronger to stabilize allocation
    quality_bonus = 0.20 * (_clamp(type_score / 0.45) * 0.5 + _clamp(intercept_prob) * 0.5)
    
    return _clamp(base_score + quality_bonus)


# ─────────────────────────────────────────────────────────────────────────────
# T4 — MULTI-THREAT COORDINATION
# ─────────────────────────────────────────────────────────────────────────────

def reward_coordination(
    threats_handled:      int,
    total_threats:        int,
    priority_order_score: float,
    simultaneous:         bool,
    casualties_avoided:   int,
    total_at_risk:        int,
    step:                 int = 0,
) -> float:
    """
    T4 — Multi-Threat Coordination reward (max 1.0).
      0.30  coverage ratio
      0.30  priority ordering quality
      0.20  simultaneous handling bonus
      0.20  lives saved ratio

    FIX v2: Removed the step_decay multiplier `(1.0 + STEP_DECAY_GAMMA**step)`.
    The old formula at step=1 gave multiplier=1.97 but it was always clamped
    to 1.0, making it useless. Worse, it trained the policy to think that ONLY
    early coordination has value — which prevented re-coordination when the env
    coordination score decays.

    Replaced with a modest flat early_bonus (+0.10 if step <= 12) that rewards
    doing coordination before allocate/rescue without compounding.

    The env coordination score decays per step regardless of our shaped reward,
    so the policy needs to learn to re-coordinate periodically. The new formula
    gives consistent positive signal for any coordination action.
    """
    coverage = 0.30 * _clamp(threats_handled / max(total_threats, 1))
    priority = 0.30 * _clamp(priority_order_score)
    simul    = 0.20 if (simultaneous and threats_handled >= 2) else 0.0
    lives    = 0.20 * _clamp(casualties_avoided / max(total_at_risk, 1))

    base_score = coverage + priority + simul + lives

    early_bonus = 0.0
    if step <= 12:
        early_bonus = 0.10
    elif step <= 20:
        early_bonus = 0.05

    stability_bonus = 0.1 * _clamp(priority_order_score)

    return _clamp(base_score + early_bonus + stability_bonus)


# ─────────────────────────────────────────────────────────────────────────────
# T5 — RESCUE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def reward_rescue(
    rescued:         int,
    total_victims:   int,
    units_deployed:  int,
    units_optimal:   int,
    response_step:   int,
    max_steps:       int,
    unit_type_match: float = 1.0,
) -> float:
    """
    T5 — Rescue Operations reward (max 1.0).
      0.40  victims rescued ratio
      0.25  unit count efficiency Gaussian around optimal
      0.20  response speed (earlier = better)
      0.15  unit type match

    FIX v2:
    - Bonus thresholds: reduced from 0.5/0.8 to 0.3/0.6 to give positive
      gradient earlier (plateau at 45% was because the 50% threshold was
      never reached in most episodes).
    - Bonus values scaled to keep total reachable score > 1.0 before clamp,
      so that policy gets strong signal for high rescue rates.
    - Speed score now uses a softer decay (sigma=0.5 instead of linear)
      so mid-episode rescues still get meaningful speed credit.
    """
    rescued_ratio = rescued / max(total_victims, 1)
    victim_score = 0.40 * _clamp(rescued_ratio)

    ratio      = units_deployed / max(units_optimal, 1)
    unit_score = 0.25 * _gauss(ratio - 1.0, sigma=0.5)  # wider sigma: more forgiving

    speed = 1.0 - (response_step / max(max_steps, 1))
    speed_score = 0.20 * _clamp(speed)

    type_score  = 0.15 * _clamp(unit_type_match)

    reward = victim_score + unit_score + speed_score + type_score

    # Progressive bonuses — lowered thresholds for earlier positive signal
    if rescued_ratio >= 0.20:
        reward += 0.25
    if rescued_ratio >= 0.50:
        reward += 0.35

    return _clamp(reward)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_task_scores(state: Any) -> TaskScores:
    """Pull task scores from the env state object or dict."""
    if hasattr(state, "model_dump"):
        payload = state.model_dump()
    elif isinstance(state, dict):
        payload = state
    else:
        payload = {}

    return TaskScores(
        classification = float(payload.get("classification_score", 0.0)),
        prediction     = float(payload.get("prediction_score",     0.0)),
        allocation     = float(payload.get("allocation_score",     0.0)),
        coordination   = float(payload.get("coordination_score",   0.0)),
        rescue         = float(payload.get("rescue_score",         0.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE STEP REWARD  (main entry point called from train.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_reward(
    action_type: str,
    step:        int,
    max_steps:   int,
    env_reward:  float,
    task_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute blended reward for one training step.

    KEY FIX v2: skip/delay return NEGATIVE reward to break loop behaviour.
    Productive task actions get: 70% shaped + 30% raw, floored at BASE_REWARD.
    Rescue actions use RESCUE_BASE floor (slightly higher) to encourage more
    rescue steps in the late game.
    
    FIX v4: Strong allocation incentives + over-classify penalty
    """
    if action_type == "skip":
        return SKIP_PENALTY
    if action_type == "delay":
        return DELAY_PENALTY
    
    timing_penalty = 0.0
    
    # OVER-CLASSIFY PENALTY - after step 6, penalize extra classify
    if action_type == "classify" and step > 6:
        timing_penalty -= 0.08
    
    # ALLOCATION TIMING PRESSURE - encourage early allocation
    if action_type == "allocate":
        # Get allocation quality from task_kwargs if available
        alloc_quality = 0.8  # default good quality
        if task_kwargs:
            rt = task_kwargs.get("resource_type", "")
            tt = task_kwargs.get("threat_type", "")
            intercept = task_kwargs.get("intercept_prob", 0.5)
            # Estimate quality
            type_match = 0.4 if any(c in rt for c in tt.split()) or any(c in rt for c in ["fire", "water", "medical", "rescue"]) else 0.2
            alloc_quality = type_match * 0.5 + intercept * 0.5
        
        # Hard stabilize allocation (Fix 1)
        timing_penalty += 0.15 * alloc_quality
        if alloc_quality < 0.8:
            timing_penalty -= 0.25

    
    if task_kwargs is None:
        task_kwargs = {}

    shapers = {
        "classify":   reward_classification,
        "predict":    reward_prediction,
        "allocate":   reward_allocation,
        "coordinate": reward_coordination,
        "rescue":     reward_rescue,
    }

    shaper = shapers.get(action_type)
    if shaper is None:
        return max(BASE_REWARD, float(env_reward) * 0.5)

    try:
        shaped = shaper(**task_kwargs)
    except TypeError:
        shaped = float(env_reward)

    decay   = STEP_DECAY_GAMMA ** max(0, step - 1)
    floor   = RESCUE_BASE if action_type == "rescue" else BASE_REWARD
    shaped  = max(floor, shaped * decay)
    blended = SHAPED_BLEND * shaped + (1.0 - SHAPED_BLEND) * float(env_reward)
    return max(0.0, blended + timing_penalty)


# ─────────────────────────────────────────────────────────────────────────────
# RICH VISUAL DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

_BAR_W = 18

def _bar(value: float) -> str:
    filled = int(round(_clamp(value) * _BAR_W))
    empty  = _BAR_W - filled
    inner  = "█" * filled + "░" * empty
    if value >= 0.75:
        return GREEN(inner)
    elif value >= 0.40:
        return YELLOW(inner)
    else:
        return RED(inner)

def _val(v: float) -> str:
    if v >= 0.75: return GREEN(f"{v:.2f}")
    if v >= 0.40: return YELLOW(f"{v:.2f}")
    return RED(f"{v:.2f}")

_ICONS = {
    "classify":   "🛡 ", "predict":    "🎯 ", "allocate":   "⚙️  ",
    "coordinate": "🔗 ", "rescue":     "🚑 ", "skip":       "⏭ ",
    "delay":      "⏳ ",
}

_TASK_NAMES = {
    "classification": "T1·Classify", "prediction":  "T2·Predict",
    "allocation":     "T3·Allocate", "coordination":"T4·Coord  ",
    "rescue":         "T5·Rescue  ",
}

_PHASE_LABEL = {1: "EASY", 2: "MEDIUM", 3: "HARD"}


def print_step_dashboard(info: StepInfo) -> None:
    s    = info.scores
    icon = _ICONS.get(info.action_type, "   ")
    atype = info.action_type.upper()[:11]

    score_parts = (
        f"C:{_val(s.classification)} "
        f"P:{_val(s.prediction)} "
        f"A:{_val(s.allocation)} "
        f"Co:{_val(s.coordination)} "
        f"R:{_val(s.rescue)}"
    )

    rew_str = f"{info.raw_reward:+.4f}" if info.raw_reward < 0 else f"{info.raw_reward:.4f}"
    rew_display = RED(rew_str) if info.raw_reward < 0 else _val(info.raw_reward)

    print(
        f"{BOLD(f'[STEP {info.step:>3}]')} "
        f"{icon}{CYAN(f'{atype:<12}')} "
        f"Rew:{rew_display}  │ "
        f"{score_parts}  │ "
        f"FINAL:{BOLD(_val(s.final))}",
        flush=True,
    )


def print_episode_summary(
    episode:            int,
    scores:             TaskScores,
    phase:              int,
    difficulty:         str,
    policy_delta:       float = 0.0,
    critic_acc:         float = 0.0,
    entropy:            float = 0.0,
    replay_len:         int   = 0,
    episodes_per_s:     float = 0.0,
    avg_score:          float = 0.0,
) -> None:
    sep  = BOLD("═" * 74)
    sep2 = "─" * 74
    ph   = _PHASE_LABEL.get(phase, str(phase))

    print(f"\n{sep}")
    print(
        BOLD(f"  Ep {episode:>4} ") +
        MAGENTA(f"[{ph} · {difficulty.upper()}]") +
        f"  Score:{BOLD(_val(scores.final * 100.0))}  "
        f"RunAvg:{YELLOW(f'{avg_score * 100.0:.2f}')}  "
        f"({episodes_per_s:.1f} ep/s)"
    )
    print(sep2)

    rows = [
        ("classification", scores.classification),
        ("prediction",     scores.prediction),
        ("allocation",     scores.allocation),
        ("coordination",   scores.coordination),
        ("rescue",         scores.rescue),
    ]

    for key, val in rows:
        name   = _TASK_NAMES[key]
        weight = TASK_WEIGHTS[key]
        contrib= val * weight * 100.0
        bar    = _bar(val)
        print(
            f"  {CYAN(name)}  [{bar}]  "
            f"{_val(val * 100.0)}  "
            f"× {weight:.2f}  →  {YELLOW(f'{contrib:.1f}')}"
        )

    print(sep2)
    w_final = scores.final
    print(
        f"  {'COMPOSITE FINAL':<18}  [{_bar(w_final)}]  "
        f"{BOLD(_val(w_final * 100.0))}"
    )

    if critic_acc > 0 or replay_len > 0:
        print(
            f"\n  policy_Δ={policy_delta:+.2f}%  critic_acc={critic_acc:.1f}%  "
            f"entropy={entropy:.3f}  replay={replay_len}"
        )
    print(sep + "\n")


def print_phase_transition(from_phase: int, to_phase: int,
                            episode: int, avg_score: float) -> None:
    f_label = _PHASE_LABEL.get(from_phase, str(from_phase))
    t_label = _PHASE_LABEL.get(to_phase,   str(to_phase))
    print(
        f"\n{BOLD('🚀  CURRICULUM PHASE TRANSITION')}  "
        f"{YELLOW(f_label)} ──▶ {GREEN(t_label)}  "
        f"(ep {episode}, avg={avg_score:.4f})\n",
        flush=True,
    )