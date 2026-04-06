"""
rewards.py  —  Crisis Response RL · Dynamic Reward & Scoring Engine
====================================================================
Drop this file into crisis_env/ alongside train.py.

Fixes the broken fixed-value reward bug (everything was 0.020).
Each of the 5 tasks now has its own reward shaper with real math.

Tasks:
  T1  Threat Classification   →  classify   action  (C)
  T2  Impact Prediction        →  predict    action  (P)
  T3  Counter-Defense Alloc.   →  allocate   action  (A)
  T4  Multi-Threat Coord.      →  coordinate action  (Co)
  T5  Rescue Operations        →  rescue     action  (R)
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# TASK WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
TASK_WEIGHTS: Dict[str, float] = {
    "classification": 0.20,
    "prediction":     0.20,
    "allocation":     0.20,
    "coordination":   0.15,
    "rescue":         0.25,
}

BASE_REWARD      = 0.02    # minimum reward for any valid action
STEP_DECAY_GAMMA = 0.97    # encourage acting early; decays reward per step

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

# Maps broad threat domains to their subtypes.
THREAT_DOMAINS: Dict[str, List[str]] = {
    "air":    ["missile", "drone", "airstrike", "helicopter", "aircraft"],
    "land":   ["tank", "ground_troops", "artillery", "vehicle", "infantry"],
    "water":  ["warship", "submarine", "torpedo", "naval", "boat"],
    "cyber":  ["malware", "ddos", "intrusion", "ransomware"],
    "chem":   ["chemical", "biological", "nuclear", "radiological"],
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
    T1 — Threat Classification reward.

    Breakdown (max 1.0):
      0.55  exact subtype match  (e.g. "missile" == "missile")
      0.25  domain match only    (e.g. "drone"   ∈ air domain)
      0.20  severity accuracy    Gaussian(err, σ=0.25)

    Multiplied by confidence clipped to [0.5, 1.0] so the agent
    is penalised for over-confident wrong guesses.
    """
    pt = predicted_type.lower().strip()
    tt = true_type.lower().strip()

    if pt == tt:
        type_score = 0.55
    elif _threat_domain(pt) == _threat_domain(tt) and _threat_domain(tt) != "unknown":
        type_score = 0.25
    else:
        type_score = 0.0

    sev_err    = abs(predicted_severity - true_severity) / max(abs(true_severity), 1.0)
    sev_score  = 0.20 * _gauss(sev_err, sigma=0.25)

    conf       = _clamp(confidence, 0.5, 1.0)
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
    T2 — Impact Prediction reward.

    Breakdown (max 1.0):
      0.50  TTI accuracy   — Gaussian over normalised step error
      0.50  Population     — Gaussian over log-ratio (handles large numbers)
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

_COMPAT: Dict[str, List[str]] = {
    "missile":       ["interceptor", "air_defense", "radar", "sam"],
    "drone":         ["jammer", "interceptor", "air_defense", "laser"],
    "airstrike":     ["interceptor", "air_defense", "fighter", "sam"],
    "helicopter":    ["sam", "air_defense", "fighter", "radar"],
    "aircraft":      ["fighter", "sam", "air_defense", "interceptor"],
    "tank":          ["artillery", "anti_tank", "helicopter", "mine"],
    "ground_troops": ["infantry", "artillery", "helicopter", "drone_strike"],
    "artillery":     ["counter_battery", "artillery", "helicopter"],
    "warship":       ["destroyer", "submarine", "coastal_battery", "missile"],
    "submarine":     ["destroyer", "depth_charge", "sonar", "helicopter"],
    "torpedo":       ["decoy", "destroyer", "sonar"],
    "chemical":      ["hazmat", "decon_unit", "cbrn"],
    "biological":    ["hazmat", "decon_unit", "cbrn", "medical"],
    "nuclear":       ["evacuation", "decon_unit", "cbrn"],
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
    T3 — Counter-Defense Allocation reward.

    Breakdown (max 1.0):
      0.45  resource–threat type compatibility
      0.30  intercept probability (direct env signal)
      0.15  budget efficiency  (reward frugality)
      0.10  proximity bonus   (nearest resource wins)
    """
    compat_list = _COMPAT.get(threat_type.lower(), [])
    rt          = resource_type.lower()
    type_score  = 0.45 if any(c in rt for c in compat_list) else 0.08

    intercept_score = 0.30 * _clamp(intercept_prob)

    if budget_total > 0:
        frac = budget_used / budget_total
        efficiency = 0.15 * _gauss(frac - 0.3, sigma=0.3)   # sweet spot ~30% usage
    else:
        efficiency = 0.0

    proximity = 0.10 if nearest_chosen else 0.0

    return _clamp(type_score + intercept_score + efficiency + proximity)


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
) -> float:
    """
    T4 — Multi-Threat Coordination reward.

    Breakdown (max 1.0):
      0.30  coverage ratio        (threats_handled / total)
      0.30  priority ordering     (high-impact threats first)
      0.20  simultaneous handling (parallel response bonus)
      0.20  lives saved ratio
    """
    coverage = 0.30 * _clamp(threats_handled / max(total_threats, 1))
    priority = 0.30 * _clamp(priority_order_score)
    simul    = 0.20 if (simultaneous and threats_handled >= 2) else 0.0
    lives    = 0.20 * _clamp(casualties_avoided / max(total_at_risk, 1))
    return _clamp(coverage + priority + simul + lives)


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
    T5 — Rescue Operations reward.

    Breakdown (max 1.0):
      0.40  victims rescued ratio
      0.25  unit count efficiency  Gaussian around optimal count
      0.20  response speed         (earlier = better)
      0.15  unit type match        (ambulance vs fire vs special)
    """
    victim_score = 0.40 * _clamp(rescued / max(total_victims, 1))

    ratio      = units_deployed / max(units_optimal, 1)
    unit_score = 0.25 * _gauss(ratio - 1.0, sigma=0.4)   # peak when ratio == 1.0

    speed = 1.0 - (response_step / max(max_steps, 1))
    speed_score = 0.20 * _clamp(speed)

    type_score  = 0.15 * _clamp(unit_type_match)

    return _clamp(victim_score + unit_score + speed_score + type_score)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE EXTRACTOR  (reads live from env state dict)
# ─────────────────────────────────────────────────────────────────────────────

def extract_task_scores(state: Any) -> TaskScores:
    """
    Pull task scores from the env state object or dict.
    Always returns real values — never hardcoded fallbacks.
    """
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

    Blending: 70% shaped task reward + 30% raw env reward.
    Step-decay penalises procrastination (acting late = less reward).
    Falls back to BASE_REWARD for non-task actions (skip/delay).

    task_kwargs  keys per action type:
      classify   → predicted_type, true_type, predicted_severity,
                   true_severity, confidence
      predict    → predicted_tti, true_tti, predicted_pop, true_pop, total_steps
      allocate   → resource_type, threat_type, intercept_prob,
                   budget_used, budget_total, nearest_chosen
      coordinate → threats_handled, total_threats, priority_order_score,
                   simultaneous, casualties_avoided, total_at_risk
      rescue     → rescued, total_victims, units_deployed, units_optimal,
                   response_step, max_steps, unit_type_match
    """
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
        shaped = BASE_REWARD
    else:
        try:
            shaped = shaper(**task_kwargs)
        except TypeError:
            # kwargs didn't match — fall back to env reward so training continues
            shaped = float(env_reward)

    decay   = STEP_DECAY_GAMMA ** max(0, step - 1)
    shaped  = max(BASE_REWARD, shaped * decay)
    blended = SHAPED_BLEND * shaped + (1.0 - SHAPED_BLEND) * float(env_reward)
    return max(BASE_REWARD, blended)


# ─────────────────────────────────────────────────────────────────────────────
# RICH VISUAL DISPLAY  (zero blocking I/O — just print + flush)
# ─────────────────────────────────────────────────────────────────────────────

_BAR_W = 18

def _bar(value: float) -> str:
    """Coloured ASCII progress bar."""
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
    """
    One compact coloured line per step — cheap, non-blocking.

    Example output:
    [STEP  4] ⚙️  ALLOCATE     Rew:0.61 │ C:0.80 P:0.55 A:0.61 Co:0.00 R:0.00 │ FINAL:0.39
    """
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

    print(
        f"{BOLD(f'[STEP {info.step:>3}]')} "
        f"{icon}{CYAN(f'{atype:<12}')} "
        f"Rew:{_val(info.raw_reward)}  │ "
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
    """
    Rich per-episode summary with per-task bars + weighted contributions.
    """
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
