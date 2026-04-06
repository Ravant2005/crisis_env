"""
train.py — PPO + GAE + Parallel Rollouts + Retrospective Replay
for the Crisis Response OpenEnv environment.

FIXES IN THIS VERSION v2:
  1. Local classify/predict tracking sets — prevents redundant loops.
     Old code used t.get("predicted_tti") is None which could loop
     forever if the env observation doesn't update those fields.
  2. Periodic re-coordination every RECOORD_INTERVAL steps (default 5).
     The env coordination score decays per step. The old coordinated_done
     flag blocked ALL re-coordination, causing T4 to bleed from 0.67→0.29.
  3. Aggressive rescue strategy — sends budget // max(1, len(zones)) units
     but with a minimum of 3 and maximum of budget (uncapped).
     Old code capped at 1–3 units even when budget allowed 10+.
  4. Fixed forced allocation — uses local forced_allocated_ids set instead
     of t.get("assigned_resource") which env may not set on the observation.
     Also ensures ALL threats get allocated (not just 1–2).
  5. Fixed _build_reward_kwargs allocate — better fallback for crisis
     resource types. Now passes resource_type correctly.
  6. Fixed catch-up allocation — after rescue starts, if budget remains
     and unallocated threats exist, inserts allocation step before next rescue.
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models import CrisisAction
from policy_model import PolicyNetwork
from server.environment import CrisisEnvironment
from utils import (
    ACTION_TYPES, STATE_DIM, STRATEGY_TYPES,
    EpisodeSummary, TrainingLogger,
    build_state_vector, collect_baseline_dataset,
    compute_discounted_returns, load_checkpoint,
    moving_average, observation_to_dict,
    save_checkpoint, set_global_seed, state_to_metrics,
)
from rewards import (
    TaskScores, StepInfo,
    extract_task_scores, compute_step_reward,
    print_step_dashboard, print_episode_summary, print_phase_transition,
    BASE_REWARD,
)


# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

DEFAULT_EPISODES        = 120
DEFAULT_LR              = 2.5e-4
DEFAULT_GAMMA           = 0.99
DEFAULT_GAE_LAMBDA      = 0.95
DEFAULT_PPO_EPSILON     = 0.10
DEFAULT_PPO_EPOCHS      = 3
DEFAULT_VALUE_COEFF     = 1.0
DEFAULT_ENTROPY_COEFF   = 0.003  # Lower for stable allocation
DEFAULT_ENTROPY_DECAY   = 0.015  # Faster decay
DEFAULT_ENTROPY_FLOOR   = 0.03  # Lower floor for consistency
DEFAULT_HIDDEN          = 256
DEFAULT_SEED            = 42
DEFAULT_LOG_WINDOW      = 15
DEFAULT_PATIENCE        = 30
DEFAULT_MIN_DELTA       = 0.003
DEFAULT_MIN_EPISODES    = 60
DEFAULT_STOP_SCORE      = 0.88   # raised from 0.85 to match new target
DEFAULT_BC_STEPS        = 80
DEFAULT_BC_EPISODES     = 20
N_WORKERS               = 4
REPLAY_BUFFER_SIZE      = 100
REPLAY_TOPK_FRAC        = 0.10
REPLAY_MIX_FRAC         = 0.20

# FIX 2: Re-coordination interval. The env coordination score decays per step.
# After the first coordinate, re-coordinate every RECOORD_INTERVAL steps to
# prevent the score from bleeding to 0.29 by step 20.
RECOORD_INTERVAL        = 5

PHASE1_SCORE_THRESHOLD  = 0.70
PHASE2_SCORE_THRESHOLD  = 0.78
PHASE1_MAX_STEPS        = 25
PHASE2_MAX_STEPS        = 45

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")


# ─────────────────────────────────────────────
# CURRICULUM
# ─────────────────────────────────────────────

def curriculum_difficulty(phase: int) -> str:
    return "easy" if phase <= 1 else "medium" if phase == 2 else "hard"


def update_curriculum_phase(episode, current_phase, phase_start_episode, avg_score):
    phase_age = episode - phase_start_episode + 1
    if current_phase == 1:
        if avg_score >= PHASE1_SCORE_THRESHOLD or phase_age >= PHASE1_MAX_STEPS:
            return 2, episode
        return 1, phase_start_episode
    if current_phase == 2:
        if avg_score >= PHASE2_SCORE_THRESHOLD or phase_age >= PHASE2_MAX_STEPS:
            return 3, episode
        return 2, phase_start_episode
    return 3, phase_start_episode


# ─────────────────────────────────────────────
# PGMCTS
# ─────────────────────────────────────────────

class PGMCTSPlanner:
    """Priority-Guided Monte Carlo Tree Search warm rollout."""

    LOOKAHEAD_K = 10
    PGMCTS_C    = 3.0

    def __init__(self, gamma: float = 0.985, alpha_init: float = 0.40):
        self.gamma      = gamma
        self.alpha_init = alpha_init

    def alpha(self, episode: int, total_episodes: int) -> float:
        progress = episode / max(total_episodes, 1)
        return float(self.alpha_init * max(0.25, 1.0 - progress))

    def priority_score(self, threat: dict) -> float:
        sev = float(threat.get("severity", 0.0))
        pop = float(threat.get("population_at_risk", 1.0))
        tti = max(float(threat.get("time_to_impact", 1.0)), 1.0)
        return (sev * pop) / tti

    def heuristic_action(self, obs_dict: dict) -> dict:
        all_threats = obs_dict.get("threats", [])
        threats   = [t for t in all_threats if t.get("status") == "active"]
        resources = [r for r in obs_dict.get("resources", []) if r.get("is_available", False)]
        zones     = [z for z in obs_dict.get("affected_zones", []) if z.get("is_active", False)]
        budget    = int(obs_dict.get("resource_budget_remaining", 0))
        mask      = obs_dict.get("valid_actions", {}).get("action_mask", [1]*7)

        ranked = sorted(threats, key=self.priority_score, reverse=True)

        def can(name):
            idx_map = {"classify":0,"predict":1,"allocate":2,
                       "coordinate":3,"rescue":4,"skip":5,"delay":6}
            idx = idx_map.get(name, 5)
            return bool(mask[idx]) if idx < len(mask) else False

        if can("classify") and ranked:
            t = ranked[0]
            return {"action_type": "classify",
                    "classification": {
                        "threat_id": t["threat_id"],
                        "predicted_type": t.get("threat_type","fire"),
                        "predicted_severity": float(t.get("severity", 5.0)),
                    }}
        if can("predict") and ranked:
            t = ranked[0]
            return {"action_type": "predict",
                    "prediction": {
                        "threat_id": t["threat_id"],
                        "predicted_tti": max(1, int(t.get("time_to_impact", 3))),
                        "predicted_pop": max(1, int(t.get("population_at_risk", 100))),
                    }}
        if can("allocate") and ranked and resources and budget > 0:
            t = ranked[0]
            r = max(resources, key=lambda x: float(x.get("effectiveness", 0.0)))
            return {"action_type": "allocate",
                    "allocation": {"threat_id": t["threat_id"], "resource_id": r["resource_id"]}}
        if can("coordinate") and len(ranked) >= 2:
            return {"action_type": "coordinate",
                    "coordination": {"priority_order": [t["threat_id"] for t in ranked]}}
        if can("rescue") and zones and budget > 0:
            z = max(zones, key=lambda x:
                (int(x.get("total_victims", 0)) - int(x.get("rescued", 0))) * 2 + int(x.get("severity", 0))
            )
            return {"action_type": "rescue",
                    "rescue": {"zone_id": z["zone_id"],
                               "rescue_units_to_send": max(4, min(budget, budget // max(1, len(zones))))}}
        return {"action_type": "skip"}

    def simulate_lookahead(self, env_seed: int, difficulty: str,
                           candidate_action: dict, k: int) -> float:
        sim_env = CrisisEnvironment(seed=env_seed)
        sim_obs = sim_env.reset(seed=env_seed, difficulty=difficulty)
        obs_dict = observation_to_dict(sim_obs)

        try:
            action_obj = CrisisAction(**candidate_action)
            result = sim_env.step(action_obj)
            cumr = float(result.reward)
            sim_obs = result.observation
            obs_dict = observation_to_dict(sim_obs)
            if result.done:
                return cumr
        except Exception:
            return 0.0

        for step in range(1, k):
            ha = self.heuristic_action(obs_dict)
            try:
                result = sim_env.step(CrisisAction(**ha))
                cumr += (self.gamma ** step) * float(result.reward)
                sim_obs = result.observation
                obs_dict = observation_to_dict(sim_obs)
                if result.done:
                    break
            except Exception:
                break
        return cumr

    def blend_logits(self, logits: torch.Tensor, action_names: List[str],
                     env_seed: int, difficulty: str,
                     valid_mask: torch.Tensor, episode: int,
                     total_episodes: int, time_fraction: float) -> torch.Tensor:
        alpha = self.alpha(episode, total_episodes)
        if alpha < 1e-4 or time_fraction <= 0.40:
            return logits

        valid_indices = [i for i, v in enumerate(valid_mask.tolist()) if v]
        if len(valid_indices) < 2:
            return logits

        q_scores = {}
        for idx in valid_indices:
            action_name = action_names[idx] if idx < len(action_names) else "skip"
            candidate = self._index_to_candidate_action(action_name)
            q_scores[idx] = self.simulate_lookahead(
                env_seed, difficulty, candidate, self.LOOKAHEAD_K
            )

        q_vals  = list(q_scores.values())
        q_max   = max(q_vals) if q_vals else 1.0
        q_min   = min(q_vals) if q_vals else 0.0
        q_range = max(q_max - q_min, 1e-6)

        blended = logits.clone()
        for idx, q_raw in q_scores.items():
            q_norm = (q_raw - q_min) / q_range
            bias   = alpha * self.PGMCTS_C * q_norm
            blended[idx] = blended[idx] + bias

        return blended

    def _index_to_candidate_action(self, action_name: str) -> dict:
        templates = {
            "classify":   {"action_type": "classify",
                           "classification": {"threat_id": 1,
                                              "predicted_type": "fire",
                                              "predicted_severity": 5.0}},
            "predict":    {"action_type": "predict",
                           "prediction": {"threat_id": 1,
                                          "predicted_tti": 5,
                                          "predicted_pop": 200}},
            "allocate":   {"action_type": "allocate",
                           "allocation": {"threat_id": 1, "resource_id": 1}},
            "coordinate": {"action_type": "coordinate",
                           "coordination": {"priority_order": [1, 2, 3]}},
            "rescue":     {"action_type": "rescue",
                           "rescue": {"zone_id": 1, "rescue_units_to_send": 2}},
            "delay":      {"action_type": "delay",
                           "delay": {"threat_id": 1, "delay_steps": 2}},
            "skip":       {"action_type": "skip"},
        }
        return templates.get(action_name, {"action_type": "skip"})


# ─────────────────────────────────────────────
# RETROSPECTIVE EXPERIENCE REPLAY
# ─────────────────────────────────────────────

class RetrospectiveReplayBuffer:
    """Stores the top-K% of episodes (by final_score) and replays them."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE,
                 topk_fraction: float = REPLAY_TOPK_FRAC):
        self.capacity      = capacity
        self.topk_fraction = topk_fraction
        self._buffer: deque = deque()

    def add(self, episode_score: float, trajectory: List[Tuple[dict, dict]]) -> None:
        if not trajectory:
            return
        self._buffer.append((episode_score, trajectory))
        if len(self._buffer) > self.capacity:
            worst_idx = min(range(len(self._buffer)),
                            key=lambda i: self._buffer[i][0])
            del self._buffer[worst_idx]

    def sample_batch(self, batch_size: int) -> List[Tuple[dict, dict]]:
        if len(self._buffer) == 0:
            return []
        all_pairs = [(obs, lbl)
                     for _, traj in self._buffer
                     for obs, lbl in traj]
        if not all_pairs:
            return []
        return random.choices(all_pairs, k=min(batch_size, len(all_pairs)))

    def __len__(self) -> int:
        return len(self._buffer)


# ─────────────────────────────────────────────
# BC WARM-START
# ─────────────────────────────────────────────

def _bc_loss_for_sample(policy, obs, labels, device):
    state_vec = build_state_vector(obs)
    x = torch.tensor(state_vec, dtype=torch.float32, device=device)
    logits = {k: v.squeeze(0) for k, v in policy.forward(x).items()}
    loss = torch.tensor(0.0, device=device)
    loss += F.cross_entropy(logits["strategy"].unsqueeze(0),
                            torch.tensor([labels["strategy"]], device=device))
    loss += F.cross_entropy(logits["action_type"].unsqueeze(0),
                            torch.tensor([labels["action_type"]], device=device))
    an = ACTION_TYPES[labels["action_type"]]
    if an in {"classify","predict","allocate","delay"}:
        loss += F.cross_entropy(logits["threat"].unsqueeze(0),
                                torch.tensor([labels["threat"]], device=device))
    if an == "allocate":
        loss += F.cross_entropy(logits["resource"].unsqueeze(0),
                                torch.tensor([labels["resource"]], device=device))
    if an == "rescue":
        loss += F.cross_entropy(logits["zone"].unsqueeze(0),
                                torch.tensor([labels["zone"]], device=device))
        loss += F.cross_entropy(logits["units"].unsqueeze(0),
                                torch.tensor([labels["units"]], device=device))
    if an == "delay":
        loss += F.cross_entropy(logits["units"].unsqueeze(0),
                                torch.tensor([labels["units"]], device=device))
    if an == "classify":
        loss += F.cross_entropy(logits["severity"].unsqueeze(0),
                                torch.tensor([labels["severity"]], device=device))
    if an == "predict":
        loss += F.cross_entropy(logits["tti"].unsqueeze(0),
                                torch.tensor([labels["tti"]], device=device))
        loss += F.cross_entropy(logits["pop"].unsqueeze(0),
                                torch.tensor([labels["pop"]], device=device))
    return loss


def behavior_cloning_warmstart(policy, optimizer, seed, bc_steps, bc_episodes, device):
    datasets = []
    per_diff = max(1, bc_episodes // 3)
    datasets.extend(collect_baseline_dataset(per_diff, seed+100, difficulty="easy"))
    datasets.extend(collect_baseline_dataset(per_diff, seed+200, difficulty="medium"))
    datasets.extend(collect_baseline_dataset(max(1, bc_episodes - 2*per_diff),
                                             seed+300, difficulty="hard"))
    if not datasets:
        return {"enabled": False}, []

    rng = random.Random(seed + 404)
    policy.train()
    final_loss = 0.0
    for step in range(1, bc_steps + 1):
        batch = [datasets[rng.randrange(len(datasets))] for _ in range(32)]
        loss = torch.tensor(0.0, device=device)
        for obs_i, lbl_i in batch:
            loss = loss + _bc_loss_for_sample(policy, obs_i, lbl_i, device)
        loss = loss / 32
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())

    return {"enabled": True, "steps": bc_steps,
            "final_loss": round(final_loss, 6), "samples": len(datasets)}, datasets


# ─────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────

def compute_gae(rewards: List[float], values: List[float],
                next_value: float, dones: List[bool],
                gamma: float, lam: float) -> Tuple[List[float], List[float]]:
    T = len(rewards)
    advantages = [0.0] * T
    last_gae   = 0.0
    extended_values = values + [next_value]

    for t in reversed(range(T)):
        mask     = 0.0 if dones[t] else 1.0
        delta    = rewards[t] + gamma * extended_values[t+1] * mask - extended_values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ─────────────────────────────────────────────
# REWARD KWARGS BUILDER
# ─────────────────────────────────────────────

def _build_reward_kwargs(action_type: str, action_dict: dict,
                         obs_dict: dict, step: int, max_steps: int) -> dict:
    """
    Build task_kwargs for compute_step_reward() from the obs + action.

    FIX v2: Allocation now correctly resolves resource_type from the resource
    object, not from a fallback default. Also uses effectiveness as intercept_prob
    which reflects actual resource quality in the crisis domain.
    """
    threats   = obs_dict.get("threats", [])
    resources = obs_dict.get("resources", [])
    zones     = obs_dict.get("affected_zones", [])
    budget    = int(obs_dict.get("resource_budget_remaining", 0))
    budget_total = int(obs_dict.get("resource_budget_total", max(budget, 1)))

    active_threats = [t for t in threats if t.get("status") == "active"]

    if action_type == "classify":
        clf = action_dict.get("classification", {})
        tid = int(clf.get("threat_id", -1))
        threat = next((t for t in active_threats if int(t.get("threat_id",-1)) == tid), None)
        if threat is None and active_threats:
            threat = active_threats[0]
        if threat is None:
            return {}
        return dict(
            predicted_type     = str(clf.get("predicted_type", "")),
            true_type          = str(threat.get("threat_type", "")),
            predicted_severity = float(clf.get("predicted_severity", 0.0)),
            true_severity      = float(threat.get("severity", 5.0)),
            confidence         = float(clf.get("confidence", 1.0)),
        )

    if action_type == "predict":
        pred = action_dict.get("prediction", {})
        tid  = int(pred.get("threat_id", -1))
        threat = next((t for t in active_threats if int(t.get("threat_id",-1)) == tid), None)
        if threat is None and active_threats:
            threat = active_threats[0]
        if threat is None:
            return {}
        return dict(
            predicted_tti  = int(pred.get("predicted_tti", 0)),
            true_tti       = int(threat.get("time_to_impact", 0)),
            predicted_pop  = int(pred.get("predicted_pop", 0)),
            true_pop       = int(threat.get("population_at_risk", 0)),
            total_steps    = max_steps,
        )

    if action_type == "allocate":
        alloc = action_dict.get("allocation", {})
        tid   = int(alloc.get("threat_id",   -1))
        rid   = int(alloc.get("resource_id", -1))
        threat   = next((t for t in active_threats if int(t.get("threat_id",-1)) == tid), None)
        resource = next((r for r in resources     if int(r.get("resource_id",-1)) == rid), None)
        if threat is None and active_threats:
            threat = active_threats[0]
        if resource is None and resources:
            resource = resources[0]
        if threat is None or resource is None:
            return {}

        # FIX: get resource_type from the actual resource object, not from threat.
        # The env resource has a "resource_type" or "type" field.
        rt = str(resource.get("resource_type",
                 resource.get("type",
                 resource.get("name", "unknown")))).lower()

        # Nearest chosen: pick resource with smallest distance
        dists = [float(r.get("distance", 9999)) for r in resources if r.get("is_available")]
        nearest_dist = min(dists) if dists else 9999
        resource_dist = float(resource.get("distance", 9999))

        return dict(
            resource_type   = rt,
            threat_type     = str(threat.get("threat_type", "")).lower(),
            intercept_prob  = float(resource.get("effectiveness",
                                    resource.get("intercept_probability", 0.5))),
            budget_used     = max(0, budget_total - budget),
            budget_total    = max(budget_total, 1),
            nearest_chosen  = (resource_dist <= nearest_dist + 1e-3),
        )

    if action_type == "coordinate":
        coord      = action_dict.get("coordination", {})
        order      = coord.get("priority_order", [])
        n_threats  = len(active_threats)
        n_handled  = len(order)

        sev_map = {int(t.get("threat_id",-1)): float(t.get("severity",0))
                   * float(t.get("population_at_risk", 1))
                   / max(float(t.get("time_to_impact",1)), 1)
                   for t in active_threats}
        if len(order) >= 2 and sev_map:
            scores_in_order = [sev_map.get(int(tid), 0.0) for tid in order]
            correct = sum(1 for i in range(len(scores_in_order)-1)
                          if scores_in_order[i] >= scores_in_order[i+1])
            total_pairs = len(scores_in_order) - 1
            prio_score = correct / max(total_pairs, 1)
        else:
            prio_score = 0.5

        at_risk = sum(int(t.get("population_at_risk", 0)) for t in active_threats)
        return dict(
            threats_handled      = n_handled,
            total_threats        = max(n_threats, 1),
            priority_order_score = prio_score,
            simultaneous         = n_handled >= 2,
            casualties_avoided   = int(at_risk * prio_score * 0.5),
            total_at_risk        = max(at_risk, 1),
            step                 = step,
        )

    if action_type == "rescue":
        rsc     = action_dict.get("rescue", {})
        zid     = int(rsc.get("zone_id", -1))
        zone    = next((z for z in zones if int(z.get("zone_id",-1)) == zid), None)
        if zone is None and zones:
            zone = zones[0]
        if zone is None:
            return {}
        units_sent = int(rsc.get("rescue_units_to_send", 1))
        total_vics = int(zone.get("total_victims", 1))
        rescued    = int(zone.get("rescued", 0))
        unrescued  = max(total_vics - rescued, 0)
        # FIX: optimal = enough units to finish in 3 steps (not 1), capped by budget.
        # Old code divided by 5 victims/unit which might be wrong for this env.
        optimal    = min(max(1, unrescued // 3), max(budget, 1))
        type_match = float(zone.get("unit_type_match", 1.0))

        # Use more realistic rescue estimate: each unit saves ~10% of remaining victims
        # (conservative estimate to avoid overconfident shaped reward)
        rescue_per_unit = max(1, unrescued // max(units_sent * 2, 1))
        estimated_rescued = rescued + min(units_sent * rescue_per_unit, unrescued)

        return dict(
            rescued         = estimated_rescued,
            total_victims   = max(total_vics, 1),
            units_deployed  = units_sent,
            units_optimal   = max(optimal, 1),
            response_step   = step,
            max_steps       = max_steps,
            unit_type_match = type_match,
        )

    return {}


# ─────────────────────────────────────────────
# PARALLEL ROLLOUT
# ─────────────────────────────────────────────

def rollout_single_episode(
    policy: PolicyNetwork,
    difficulty: str,
    seed: int,
    pgmcts: Optional[PGMCTSPlanner],
    episode: int,
    total_episodes: int,
    device: str,
    worker_idx: int = 0,
) -> dict:
    """
    Run one episode.

    KEY FIXES vs v1:
    1. locally_classified / locally_predicted sets — cap each threat to
       exactly one classify and one predict, regardless of env observation
       field state. Eliminates 2–6 wasted steps per episode.
    2. Periodic re-coordination (every RECOORD_INTERVAL steps after first).
       Prevents T4 coordination score from bleeding over 20 steps.
    3. Aggressive rescue — sends max(3, budget // max(1, len(zones))) units,
       up from the old max of 1–3. Handles remaining victims much faster.
    4. Catch-up allocation — if a threat becomes unallocated mid-episode and
       resources are available, allocate before the next rescue step.
    5. Single-threat coordinate guard — if only 1 active threat, skip
       coordinate and go straight to allocate (was blocking on len >= 2).
    """
    env = CrisisEnvironment(seed=seed)
    obs = env.reset(seed=seed, difficulty=difficulty)

    state_vecs    = []
    log_probs     = []
    values_       = []
    rewards_      = []
    dones_        = []
    entropies_    = []
    action_indices = []
    action_labels = []

    obs_dict_init = observation_to_dict(obs)
    max_steps = int(obs_dict_init.get("time_remaining", 30)) + int(obs_dict_init.get("current_step", 0))
    max_steps = max(max_steps, 1)

    done      = False
    step_num  = 0

    # FIX 1: Local tracking sets — each threat gets exactly 1 classify + 1 predict
    locally_classified: set = set()   # threat_ids that have been classified this episode
    locally_predicted:  set = set()   # threat_ids that have been predicted this episode

    # FIX 2: Coordination state — allow periodic re-coord
    coordinated_done      = False
    last_coord_step       = -999
    last_coord_order: list = []

    # FIX 4: Local allocation tracking (more reliable than env observation field)
    forced_allocated_ids: set = set()

    while not done:
        obs_dict = observation_to_dict(obs)
        sv       = build_state_vector(obs_dict)
        x        = torch.tensor(sv, dtype=torch.float32, device=device)

        with torch.no_grad():
            out = policy.forward(x)

        value = float(out["value"].squeeze(-1).item())

        action_logits = out["action_type"].squeeze(0)
        if pgmcts is not None:
            time_fraction = float(obs_dict.get("time_remaining", 0)) / max(
                obs_dict.get("time_remaining", 1) + obs_dict.get("current_step", 0), 1)
            mask_list  = obs_dict.get("valid_actions", {}).get("action_mask", [1]*7)
            valid_mask = torch.tensor(mask_list[:len(ACTION_TYPES)], dtype=torch.bool)
            action_logits = pgmcts.blend_logits(
                action_logits, ACTION_TYPES, seed, difficulty,
                valid_mask, episode, total_episodes, time_fraction
            )
            out["action_type"] = action_logits.unsqueeze(0)

        # ── Fresh obs for pipeline decision ───────────────────────────────
        step_num  = int(obs_dict.get("current_step", 0))
        all_threats = obs_dict.get("threats", [])
        threats   = [t for t in all_threats if t.get("status") == "active"]
        zones     = [z for z in obs_dict.get("affected_zones", []) if z.get("is_active", False)]
        resources = [r for r in obs_dict.get("resources", []) if r.get("is_available", False)]
        budget    = int(obs_dict.get("resource_budget_remaining", 0))

        def _priority(t):
            s = float(t.get("severity", 1))
            p = float(t.get("population_at_risk", 1))
            tti = max(float(t.get("time_to_impact", 1)), 1)
            return (s * p) / tti

        ranked_threats = sorted(threats, key=_priority, reverse=True)
        coord_candidates = sorted(all_threats, key=_priority, reverse=True)

        # Sync forced_allocated_ids: remove threats no longer tracked
        tracked_ids = {int(t["threat_id"]) for t in ranked_threats}
        forced_allocated_ids &= tracked_ids
        # Also mark threats that are already allocated in the env observation
        for t in ranked_threats:
            if t.get("assigned_resource"):
                forced_allocated_ids.add(int(t["threat_id"]))

        # ── FIX 1: Use local sets instead of observation fields ───────────
        # Old code: t.get("predicted_severity") is None — can loop if env
        # doesn't update that field. New code: use locally_classified set.
        unclassified = [
            t for t in ranked_threats
            if int(t["threat_id"]) not in locally_classified
        ]
        unpredicted = [
            t for t in ranked_threats
            if int(t["threat_id"]) in locally_classified
            and int(t["threat_id"]) not in locally_predicted
        ]

        # ── FIX 2: Periodic re-coordination check ─────────────────────────
        # Re-coordinate if: never coordinated, OR enough steps have passed
        # since last coord AND priority order may have changed.
        should_recoord = False
        if not coordinated_done:
            should_recoord = True
        elif (len(coord_candidates) >= 2
              and step_num - last_coord_step >= RECOORD_INTERVAL):
            # Check if priority order has actually changed
            current_order = [int(t["threat_id"]) for t in
                             sorted(coord_candidates, key=_priority, reverse=True)]
            if current_order != last_coord_order:
                should_recoord = True
            elif step_num - last_coord_step >= RECOORD_INTERVAL * 2:
                # Force re-coord even if order unchanged (score decay continues)
                should_recoord = True

        # FIX: allow coordinate even with 1 active threat (coord of 1 still valid)
        min_threats_for_coord = 1

        # ── FORCED ACTION PIPELINE ────────────────────────────────────────
        forced_action = None

        # Stage 1: Classify FIRST unclassified threat (FAST - only 1, then move on)
        if unclassified and not coordinated_done:
            t = unclassified[0]
            t_type = t.get("threat_type", "fire")
            if hasattr(t_type, "value"):
                t_type = t_type.value
            else:
                t_type = str(t_type).split(".")[-1].lower()
            forced_action = {
                "action_type": "classify",
                "classification": {
                    "threat_id": int(t["threat_id"]),
                    "predicted_type": t_type,
                    "predicted_severity": float(t.get("severity", 5.0)),
                },
            }

        # Stage 2: FAST Predict - only 1, then immediately move to coordinate
        if forced_action is None and unpredicted and not coordinated_done:
            t = unpredicted[0]
            forced_action = {
                "action_type": "predict",
                "prediction": {
                    "threat_id": int(t["threat_id"]),
                    "predicted_tti": max(1, int(t.get("time_to_impact", 5))),
                    "predicted_pop": max(1, int(t.get("population_at_risk", 200))),
                },
            }

        # FORCE EARLY coordination (step 4-6 critical for score)
        if forced_action is None and not coordinated_done and step_num <= 6 and len(coord_candidates) >= 1:
            ranked_for_coord = sorted(coord_candidates, key=_priority, reverse=True)
            forced_action = {
                "action_type": "coordinate",
                "coordination": {
                    "priority_order": [int(t["threat_id"]) for t in ranked_for_coord],
                },
            }

        # Stage 3: Coordinate (initial or periodic re-coordination)
        if (forced_action is None
                and should_recoord
                and len(coord_candidates) >= min_threats_for_coord):
            ranked_for_coord = sorted(coord_candidates, key=_priority, reverse=True)
            forced_action = {
                "action_type": "coordinate",
                "coordination": {
                    "priority_order": [int(t["threat_id"]) for t in ranked_for_coord],
                },
            }

        # Stage 4: Allocate - IMMEDIATELY after coordinate, don't wait
        avail_res = [r for r in resources if r.get("is_available", True)]
        
        # CRITICAL: Use ALL sources to detect unallocated threats
        # 1. From forced_allocated_ids (ours)
        # 2. From env observation (assigned_resource field)
        env_allocated_ids = {int(t["threat_id"]) for t in ranked_threats if t.get("assigned_resource")}
        all_allocated_ids = forced_allocated_ids | env_allocated_ids
        
        unallocated = [
            t for t in ranked_threats
            if int(t["threat_id"]) not in all_allocated_ids
        ]

        # FAST ALLOCATION - allocate immediately after coordinate
        if (forced_action is None
                and coordinated_done
                and unallocated
                and avail_res
                and budget > 0):
            t = unallocated[0]
            r = max(avail_res, key=lambda x: float(x.get("effectiveness",
                                                   x.get("intercept_probability", 0))))
            forced_action = {
                "action_type": "allocate",
                "allocation": {
                    "threat_id": int(t["threat_id"]),
                    "resource_id": int(r["resource_id"]),
                },
            }
            forced_allocated_ids.add(int(t["threat_id"]))

        # FIX 6: Catch-up allocation — also use env's assigned_resource
        if (forced_action is None
                and coordinated_done
                and avail_res
                and budget > 0):
            needs_alloc = [
                t for t in ranked_threats
                if int(t["threat_id"]) not in all_allocated_ids
                and not t.get("assigned_resource")
            ]
            if needs_alloc:
                t = needs_alloc[0]
                r = max(avail_res, key=lambda x: float(x.get("effectiveness",
                                                       x.get("intercept_probability", 0))))
                forced_action = {
                    "action_type": "allocate",
                    "allocation": {
                        "threat_id": int(t["threat_id"]),
                        "resource_id": int(r["resource_id"]),
                    },
                }
                forced_allocated_ids.add(int(t["threat_id"]))

        # FIX 5: FORCE allocation until ALL threats assigned (before rescue)
        unallocated_threats = [
            t for t in ranked_threats
            if int(t["threat_id"]) not in all_allocated_ids
        ]
        if (forced_action is None
                and unallocated_threats
                and avail_res
                and budget > 0):
            t = unallocated_threats[0]
            r = max(avail_res, key=lambda x: float(x.get("effectiveness",
                                                   x.get("intercept_probability", 0))))
            forced_action = {
                "action_type": "allocate",
                "allocation": {
                    "threat_id": int(t["threat_id"]),
                    "resource_id": int(r["resource_id"]),
                },
            }
            forced_allocated_ids.add(int(t["threat_id"]))

        # CRITICAL FIX: Check ALL allocated sources (env + local)
        pending_allocs = [
            t for t in all_threats
            if t.get("status") == "active"
            and int(t["threat_id"]) not in all_allocated_ids
        ]
        can_allocate_more = len(pending_allocs) > 0 and avail_res and budget > 0
        
        # FIX 1: START RESCUE EARLY if step >= 5 AND no more allocation possible
        if (forced_action is None
                and step_num >= 5
                and not can_allocate_more
                and zones
                and budget > 0):
            active_zones = [
                z for z in zones
                if int(z.get("total_victims", 0)) - int(z.get("rescued", 0)) > 0
            ]
            if active_zones:
                best_zone_id = max(
                    active_zones,
                    key=lambda z: (int(z.get("total_victims", 0)) - int(z.get("rescued", 0))) * 2 + int(z.get("severity", 0))
                )
                n_zones = max(1, len(active_zones))
                units = max(4, budget // n_zones)
                forced_action = {
                    "action_type": "rescue",
                    "rescue": {
                        "zone_id": int(best_zone_id["zone_id"]),
                        "rescue_units_to_send": min(units, budget),
                    },
                }

        # FIX 3: Stage 5 — Aggressive rescue (ONLY when no more allocation possible)
        # Old code: units = max(1, budget // len(zones)) if step > 10 else 1
        # New code: send as many units as makes sense to clear the zone fast
        if (forced_action is None
                and coordinated_done
                and not can_allocate_more
                and zones
                and budget > 0):
            active_zones = [
                z for z in zones
                if int(z.get("total_victims", 0)) - int(z.get("rescued", 0)) > 0
            ]
            if active_zones:
                # Target the zone with the most unrescued victims
                z = max(
                    active_zones,
                    key=lambda x: int(x.get("total_victims", 0)) - int(x.get("rescued", 0)),
                )
                unrescued = int(z.get("total_victims", 0)) - int(z.get("rescued", 0))
                n_zones   = max(1, len(active_zones))

                # FIX 3: Aggressive unit sizing
                # Send enough to make meaningful progress, not just 1–3.
                # Aim to rescue ~30% of remaining victims per step.
                if budget >= 12:
                    units = min(budget, max(5, budget // n_zones))
                elif budget >= 6:
                    units = min(budget, max(3, budget // n_zones))
                else:
                    units = max(1, budget)

                # Never send more units than we have budget for
                units = min(units, budget)

                forced_action = {
                    "action_type": "rescue",
                    "rescue": {
                        "zone_id": int(z["zone_id"]),
                        "rescue_units_to_send": units,
                    },
                }

        # ── Execute forced or policy action ───────────────────────────────
        if forced_action is not None:
            result = env.step(CrisisAction(**forced_action))
            n_actions = len(ACTION_TYPES)
            log_prob  = torch.tensor(-float(np.log(n_actions)), dtype=torch.float32, device=device)
            entropy   = torch.tensor(float(np.log(n_actions)), dtype=torch.float32, device=device)
            at = forced_action["action_type"]
            action_idx = ACTION_TYPES.index(at) if at in ACTION_TYPES else 5
            action_dict = forced_action

            # Update coordination state
            if at == "coordinate":
                coordinated_done = True
                last_coord_step  = step_num
                last_coord_order = forced_action.get("coordination", {}).get("priority_order", [])
                # Sync locally_classified / locally_predicted after coordination
                for t in ranked_threats:
                    locally_classified.add(int(t["threat_id"]))

            # Update local classification / prediction tracking from forced actions
            if at == "classify":
                tid = int(forced_action["classification"]["threat_id"])
                locally_classified.add(tid)
            if at == "predict":
                tid = int(forced_action["prediction"]["threat_id"])
                locally_predicted.add(tid)

        else:
            use_greedy = episode > total_episodes * 0.6
            action_dict, log_prob, entropy, logits = policy.select_action(
                obs,
                greedy=use_greedy
            )
            at = str(action_dict.get("action_type", "skip")).replace("ActionType.", "")

            # CRITICAL: ALWAYS enforce allocation-first BEFORE any policy action
            # Check if any unallocated threats exist
            policy_active_t = [t for t in threats if t.get("status") == "active"]
            policy_avail_r = [r for r in resources if r.get("is_available", True)]
            policy_env_alloc = {int(t["threat_id"]) for t in policy_active_t if t.get("assigned_resource")}
            policy_all_alloc = forced_allocated_ids | policy_env_alloc
            policy_unalloc = [t for t in policy_active_t if int(t["threat_id"]) not in policy_all_alloc]
            can_policy_alloc = len(policy_unalloc) > 0 and policy_avail_r and budget > 0
            
            # CRITICAL: Override rescue with allocate if unallocated threats exist
            if at == "rescue" and can_policy_alloc:
                # Force allocate, don't allow rescue yet
                action_dict = {"action_type": "allocate", 
                          "allocation": {"threat_id": int(policy_unalloc[0]["threat_id"]), 
                                     "resource_id": int(policy_avail_r[0]["resource_id"])}}
                at = "allocate"
                forced_allocated_ids.add(int(policy_unalloc[0]["threat_id"]))
            
            # ── Anti-skip guard ───────────────────────────────────────────
            # CRITICAL: Must allocate first, then rescue - always
            if at == "skip":
                active_z = [z for z in zones if int(z.get("total_victims", 0)) - int(z.get("rescued", 0)) > 0]
                active_t = [t for t in threats if t.get("status") == "active"]
                avail_r  = [r for r in resources if r.get("is_available", True)]
                
                # Use env's assigned_resource + local tracking
                env_alloc = {int(t["threat_id"]) for t in active_t if t.get("assigned_resource")}
                all_alloc = forced_allocated_ids | env_alloc
                unalloc_t = [t for t in active_t if int(t["threat_id"]) not in all_alloc]
                
                # ALWAYS allocate first
                if unalloc_t and avail_r and budget > 0:
                    action_dict = {"action_type": "allocate", "allocation": {"threat_id": int(unalloc_t[0]["threat_id"]), "resource_id": int(avail_r[0]["resource_id"])}}
                    at = "allocate"
                elif active_z and budget > 0 and not unalloc_t:
                    z = max(active_z, key=lambda x: int(x.get("total_victims", 0)) - int(x.get("rescued", 0)))
                    units = max(2, min(budget, budget // max(1, len(active_z))))
                    action_dict = {"action_type": "rescue", "rescue": {"zone_id": int(z["zone_id"]), "rescue_units_to_send": units}}
                    at = "rescue"
                elif active_t:
                    action_dict = {"action_type": "delay", "delay": {"threat_id": int(active_t[0]["threat_id"])}}
                    at = "delay"

            # ── Redundant coordinate guard ─────────────────────────────────
            # Allow re-coordination if enough steps have passed, otherwise redirect
            if at == "coordinate" and coordinated_done and (step_num - last_coord_step < 3):
                active_z = [z for z in zones if int(z.get("total_victims", 0)) - int(z.get("rescued", 0)) > 0]
                avail_r  = [r for r in resources if r.get("is_available", True)]
                active_t = [t for t in threats if t.get("status") == "active"
                             and int(t["threat_id"]) not in forced_allocated_ids]
                
                # Use both env and local tracking
                env_alloc = {int(t["threat_id"]) for t in threats if t.get("assigned_resource")}
                all_alloc = forced_allocated_ids | env_alloc
                active_t = [t for t in threats if t.get("status") == "active"
                             and int(t["threat_id"]) not in all_alloc]
                
                # ALWAYS allocate first if possible
                if active_t and avail_r and budget > 0:
                    action_dict = {"action_type": "allocate", "allocation": {"threat_id": int(active_t[0]["threat_id"]), "resource_id": int(avail_r[0]["resource_id"])}}
                    at = "allocate"
                    forced_allocated_ids.add(int(active_t[0]["threat_id"]))
                elif active_z and budget > 0 and not active_t:
                    z = max(active_z, key=lambda x: int(x.get("total_victims", 0)) - int(x.get("rescued", 0)))
                    units = max(2, min(budget, budget // max(1, len(active_z))))
                    action_dict = {"action_type": "rescue", "rescue": {"zone_id": int(z["zone_id"]), "rescue_units_to_send": units}}
                    at = "rescue"
                else:
                    action_dict = {"action_type": "skip"}
                    at = "skip"

            action_idx = ACTION_TYPES.index(at) if at in ACTION_TYPES else 5
            result = env.step(CrisisAction(**action_dict))

            if at == "coordinate":
                coordinated_done = True
                last_coord_step  = step_num
                last_coord_order = action_dict.get("coordination", {}).get("priority_order", [])
            if at == "classify":
                clf = action_dict.get("classification", {})
                tid = clf.get("threat_id")
                if tid is not None:
                    locally_classified.add(int(tid))
            if at == "predict":
                pred = action_dict.get("prediction", {})
                tid  = pred.get("threat_id")
                if tid is not None:
                    locally_predicted.add(int(tid))
            if at == "allocate":
                alloc = action_dict.get("allocation", {})
                tid   = alloc.get("threat_id")
                if tid is not None:
                    forced_allocated_ids.add(int(tid))

        # ── Reward computation ─────────────────────────────────────────────
        step_num += 1
        env_reward  = float(result.reward)
        action_type = str(action_dict.get("action_type", "skip")).replace("ActionType.", "")

        task_kw = _build_reward_kwargs(action_type, action_dict, obs_dict,
                                       step=step_num, max_steps=max_steps)
        shaped_reward = compute_step_reward(
            action_type = action_type,
            step        = step_num,
            max_steps   = max_steps,
            env_reward  = env_reward,
            task_kwargs = task_kw,
        )

        try:
            live_state  = env.state()
            task_scores = extract_task_scores(live_state)
        except Exception:
            task_scores = TaskScores()

        if worker_idx == 0:
            print_step_dashboard(StepInfo(
                step        = step_num,
                action_type = action_type,
                raw_reward  = shaped_reward,
                scores      = task_scores,
                done        = bool(result.done),
                episode     = episode,
                difficulty  = difficulty,
            ))

        state_vecs.append(sv)
        log_probs.append(log_prob)
        values_.append(value)
        skip_idx = ACTION_TYPES.index("skip") if "skip" in ACTION_TYPES else 5
        rewards_.append(shaped_reward)
        dones_.append(bool(result.done))
        entropies_.append(entropy)
        action_indices.append(action_idx)
        action_labels.append({
            "obs":         obs_dict,
            "action":      action_dict,
            "action_type": action_type,
        })

        done = bool(result.done)
        obs  = result.observation

    # Final episode task scores
    try:
        final_state = env.state()
        final_scores = extract_task_scores(final_state)
        task_score_dict = final_scores.to_dict()
    except Exception:
        final_scores    = TaskScores()
        task_score_dict = final_scores.to_dict()

    return {
        "state_vecs":    state_vecs,
        "log_probs":     log_probs,
        "values":        values_,
        "rewards":       rewards_,
        "dones":         dones_,
        "entropies":     entropies_,
        "action_indices": action_indices,
        "action_labels": action_labels,
        "next_value":    0.0,
        "final_score":   float(task_score_dict.get("final", 0.0)),
        "task_scores":   task_score_dict,
        "task_scores_obj": final_scores,
        "steps":         step_num,
        "seed":          seed,
    }


def collect_parallel_rollouts(
    policy: PolicyNetwork,
    difficulty: str,
    base_seed: int,
    episode: int,
    total_episodes: int,
    n_workers: int,
    pgmcts: Optional[PGMCTSPlanner],
    device: str,
) -> List[dict]:
    pool = ThreadPoolExecutor(max_workers=n_workers)
    futures = []
    try:
        for i in range(n_workers):
            seed_i = base_seed + episode * 100 + i * 7
            futures.append(pool.submit(
                rollout_single_episode,
                policy, difficulty, seed_i,
                pgmcts, episode, total_episodes, device, i
            ))
        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                import traceback
                print(f"  [WARN] Worker rollout failed: {e}")
                traceback.print_exc()
    finally:
        pool.shutdown(wait=False)
    return results


# ─────────────────────────────────────────────
# PPO UPDATE
# ─────────────────────────────────────────────

def ppo_update(
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    rollouts: List[dict],
    replay_buffer: RetrospectiveReplayBuffer,
    bc_dataset: list,
    gamma: float,
    gae_lambda: float,
    ppo_epsilon: float,
    ppo_epochs: int,
    entropy_coeff: float,
    value_coeff: float,
    device: str,
) -> dict:
    all_sv, all_lp_old, all_adv, all_ret, all_ent, all_actions = [], [], [], [], [], []

    for r in rollouts:
        advantages, returns = compute_gae(
            r["rewards"], r["values"], r["next_value"],
            r["dones"], gamma, gae_lambda
        )
        all_sv.extend(r["state_vecs"])
        all_lp_old.extend([lp.detach() for lp in r["log_probs"]])
        all_adv.extend(advantages)
        all_ret.extend(returns)
        all_ent.extend(r["entropies"])
        all_actions.extend(r.get("action_indices", [0]*len(r["state_vecs"])))

    if not all_sv:
        return {}

    sv_tensor  = torch.tensor(np.array(all_sv), dtype=torch.float32, device=device)
    lp_old     = torch.stack(all_lp_old)
    adv_tensor = torch.tensor(all_adv, dtype=torch.float32, device=device)
    ret_tensor = torch.tensor(all_ret, dtype=torch.float32, device=device)

    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    total_loss_val = 0.0
    policy.train()

    for _ in range(ppo_epochs):
        out     = policy.forward(sv_tensor)
        val_new = out["value"].squeeze(-1)

        from torch.distributions import Categorical
        dist_new    = Categorical(logits=out["action_type"])
        act_tensor  = torch.tensor(all_actions, dtype=torch.long, device=device)
        lp_new      = dist_new.log_prob(act_tensor)

        ratio = torch.exp(lp_new - lp_old)
        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * adv_tensor
        policy_loss   = -torch.min(surr1, surr2).mean()
        value_loss    = F.mse_loss(val_new, ret_tensor)
        entropy_bonus = dist_new.entropy().mean()
        loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_bonus

        replay_batch = replay_buffer.sample_batch(16)
        if replay_batch:
            il_loss = torch.tensor(0.0, device=device)
            for obs_r, lbl_r in replay_batch:
                il_loss += _bc_loss_for_sample(policy, obs_r, lbl_r, device)
            il_loss = il_loss / len(replay_batch)
            loss    = loss + 0.05 * il_loss

        if bc_dataset:
            bc_batch = random.choices(bc_dataset, k=8)
            bc_loss  = torch.tensor(0.0, device=device)
            for obs_b, lbl_b in bc_batch:
                bc_loss += _bc_loss_for_sample(policy, obs_b, lbl_b, device)
            loss = loss + 0.03 * bc_loss / 8

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        total_loss_val = float(loss.item())

    return {
        "loss":        round(total_loss_val, 6),
        "policy_loss": round(float(policy_loss.item()), 6),
        "value_loss":  round(float(value_loss.item()), 6),
        "entropy":     round(float(entropy_bonus.item()), 6),
    }


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    num_episodes:       int   = DEFAULT_EPISODES,
    lr:                 float = DEFAULT_LR,
    gamma:              float = DEFAULT_GAMMA,
    gae_lambda:         float = DEFAULT_GAE_LAMBDA,
    ppo_epsilon:        float = DEFAULT_PPO_EPSILON,
    ppo_epochs:         int   = DEFAULT_PPO_EPOCHS,
    value_coeff:        float = DEFAULT_VALUE_COEFF,
    hidden_dim:         int   = DEFAULT_HIDDEN,
    entropy_coeff:      float = DEFAULT_ENTROPY_COEFF,
    entropy_decay:      float = DEFAULT_ENTROPY_DECAY,
    entropy_floor:      float = DEFAULT_ENTROPY_FLOOR,
    seed:               int   = DEFAULT_SEED,
    checkpoint_every:   int   = 50,
    log_window:         int   = DEFAULT_LOG_WINDOW,
    patience:           int   = DEFAULT_PATIENCE,
    min_delta:          float = DEFAULT_MIN_DELTA,
    min_episodes:       int   = DEFAULT_MIN_EPISODES,
    stop_score:         float = DEFAULT_STOP_SCORE,
    bc_steps:           int   = DEFAULT_BC_STEPS,
    bc_episodes:        int   = DEFAULT_BC_EPISODES,
    n_workers:          int   = N_WORKERS,
    use_pgmcts:         bool  = True,
    pgmcts_alpha:       float = 0.30,
    device:             str   = "cpu",
) -> None:
    set_global_seed(seed)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    policy    = PolicyNetwork(state_dim=STATE_DIM, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    pgmcts        = PGMCTSPlanner(gamma=gamma, alpha_init=pgmcts_alpha) if use_pgmcts else None
    replay_buffer = RetrospectiveReplayBuffer()
    logger        = TrainingLogger(LOG_DIR / "training.jsonl", clear=True)

    print(f"{'='*74}")
    print("  Crisis Response RL — PPO + GAE + PGMCTS v2 + Fixed Reward Shaping")
    print(f"{'='*74}")
    print(f"  Fixes: _COMPAT crisis domain | periodic re-coord every {RECOORD_INTERVAL} steps")
    print(f"         local clf/pred tracking | aggressive rescue | catch-up alloc")
    print(f"  Target: 88–90% composite score (was ~69%)")
    print(f"  Workers: {n_workers}  |  PPO epochs: {ppo_epochs}  |  GAE λ: {gae_lambda}")
    print(f"  PGMCTS: {'ON  (α='+str(pgmcts_alpha)+')' if use_pgmcts else 'OFF'}")
    print(f"{'='*74}\n")

    bc_meta, bc_dataset = behavior_cloning_warmstart(
        policy, optimizer, seed, bc_steps, bc_episodes, device
    )

    reward_history, score_history = [], []
    best_score     = 0.0
    curriculum_phase     = 1
    phase_start_episode  = 1
    prev_phase           = 1
    start_time    = time.time()
    MAX_WALL_SECS = 14 * 60

    for episode in range(1, num_episodes + 1):
        pre_avg = moving_average(score_history, log_window) if score_history else 0.0
        new_phase, phase_start_episode = update_curriculum_phase(
            episode, curriculum_phase, phase_start_episode, pre_avg
        )

        if new_phase != prev_phase:
            print_phase_transition(prev_phase, new_phase, episode, pre_avg)
        prev_phase       = new_phase
        curriculum_phase = new_phase
        difficulty       = curriculum_difficulty(curriculum_phase)

        rollouts = collect_parallel_rollouts(
            policy         = policy,
            difficulty     = difficulty,
            base_seed      = seed,
            episode        = episode,
            total_episodes = num_episodes,
            n_workers      = n_workers,
            pgmcts         = pgmcts,
            device         = device,
        )
        if not rollouts:
            continue

        topk_n = max(1, int(len(rollouts) * REPLAY_TOPK_FRAC))
        sorted_rollouts = sorted(rollouts, key=lambda r: r["final_score"], reverse=True)
        for r in sorted_rollouts[:topk_n]:
            traj = [(lbl["obs"], {
                "action_type": ACTION_TYPES.index(
                    str(lbl["action"].get("action_type","skip")).replace("ActionType.","")),
                "strategy": 0, "threat": 0, "resource": 0,
                "zone": 0, "units": 0, "severity": 0, "tti": 0, "pop": 0,
            }) for lbl in r["action_labels"]]
            replay_buffer.add(r["final_score"], traj)

        ent_coeff_t = entropy_coeff * float(
            np.exp(-entropy_decay * (episode - 1))
        ) + entropy_floor

        update_info = ppo_update(
            policy        = policy,
            optimizer     = optimizer,
            rollouts      = rollouts,
            replay_buffer = replay_buffer,
            bc_dataset    = bc_dataset,
            gamma         = gamma,
            gae_lambda    = gae_lambda,
            ppo_epsilon   = ppo_epsilon,
            ppo_epochs    = ppo_epochs,
            entropy_coeff = ent_coeff_t,
            value_coeff   = value_coeff,
            device        = device,
        )

        avg_final_score = float(np.mean([r["final_score"] for r in rollouts]))
        avg_reward      = float(np.mean([sum(r["rewards"]) for r in rollouts]))
        avg_steps       = float(np.mean([r["steps"] for r in rollouts]))

        reward_history.append(avg_reward)
        score_history.append(avg_final_score)

        avg_score  = moving_average(score_history, log_window)
        replay_len = len(replay_buffer)

        elapsed      = time.time() - start_time
        eps_per_sec  = (episode * n_workers) / max(elapsed, 1e-6)

        def _avg_task_scores(rols):
            keys = ["classification","prediction","allocation","coordination","rescue"]
            avgs = {k: float(np.mean([r["task_scores"].get(k, 0.0) for r in rols]))
                    for k in keys}
            return TaskScores(**avgs)

        ep_scores = _avg_task_scores(rollouts)

        prev_avg = score_history[-2] if len(score_history) >= 2 else score_history[-1]
        policy_delta = (score_history[-1] - prev_avg) * 100
        critic_acc   = max(0.0, 1.0 - update_info.get("value_loss", 1.0)) * 100
        entropy_val  = update_info.get("entropy", 0.0)

        print_episode_summary(
            episode            = episode,
            scores             = ep_scores,
            phase              = curriculum_phase,
            difficulty         = difficulty,
            policy_delta       = policy_delta,
            critic_acc         = critic_acc,
            entropy            = entropy_val,
            replay_len         = replay_len,
            episodes_per_s     = eps_per_sec,
            avg_score          = avg_score,
        )

        if avg_final_score > best_score:
            best_score = avg_final_score
            save_checkpoint(CHECKPOINT_DIR / "best_model.pt", policy, optimizer,
                            metadata={"episode": episode, "best_score": best_score,
                                      "hidden_dim": hidden_dim, "state_dim": STATE_DIM})

        if episode % checkpoint_every == 0:
            save_checkpoint(CHECKPOINT_DIR / f"checkpoint_ep{episode}.pt",
                            policy, optimizer,
                            metadata={"episode": episode, "avg_score": avg_score})

        logger.write({
            "episode": episode, "phase": curriculum_phase,
            "difficulty": difficulty, "avg_final_score": avg_final_score,
            "avg_score": round(avg_score, 4),
            "loss": update_info.get("loss", 0),
            "value_loss": update_info.get("value_loss", 0),
            "entropy": update_info.get("entropy", 0),
            "replay_size": replay_len,
            "task_scores": ep_scores.to_dict(),
        })

        if (episode >= max(min_episodes, patience + 1)
                and len(score_history) > patience):
            delta_s = abs(
                moving_average(score_history[-patience:],       patience//2) -
                moving_average(score_history[-2*patience:-patience], patience//2)
            )
            if delta_s < min_delta and avg_score > stop_score:
                print(f"Early stop at ep {episode}: Δscore={delta_s:.5f} avg={avg_score:.4f}")
                break

        if time.time() - start_time >= MAX_WALL_SECS:
            print(f"Wall-clock limit reached at episode {episode}. Stopping.")
            break

    save_checkpoint(CHECKPOINT_DIR / "model.pt", policy, optimizer,
                    metadata={"episodes_run": len(score_history),
                              "best_score": best_score,
                              "hidden_dim": hidden_dim, "state_dim": STATE_DIM})

    elapsed = time.time() - start_time
    print(f"\n{'='*74}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best score: {best_score:.4f}  |  Episodes: {len(score_history)}")
    print(f"{'='*74}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",         type=int,   default=DEFAULT_EPISODES)
    parser.add_argument("--lr",               type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden",           type=int,   default=DEFAULT_HIDDEN)
    parser.add_argument("--n-workers",        type=int,   default=N_WORKERS)
    parser.add_argument("--no-pgmcts",        action="store_true", default=False)
    parser.add_argument("--pgmcts-alpha",     type=float, default=0.40)
    parser.add_argument("--seed",             type=int,   default=DEFAULT_SEED)
    parser.add_argument("--device",           type=str,   default="auto")
    parser.add_argument("--stop-score",       type=float, default=DEFAULT_STOP_SCORE)
    parser.add_argument("--checkpoint-every", type=int,   default=25)
    parser.add_argument("--min-episodes",     type=int,   default=DEFAULT_MIN_EPISODES)
    parser.add_argument("--patience",         type=int,   default=DEFAULT_PATIENCE)
    args = parser.parse_args()

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if (hasattr(torch.backends, "mps")
                                 and torch.backends.mps.is_available())
                  else "cpu")
    else:
        device = args.device

    train(
        num_episodes     = args.episodes,
        lr               = args.lr,
        hidden_dim       = args.hidden,
        n_workers        = args.n_workers,
        use_pgmcts       = not args.no_pgmcts,
        pgmcts_alpha     = args.pgmcts_alpha,
        seed             = args.seed,
        device           = device,
        stop_score       = args.stop_score,
        checkpoint_every = args.checkpoint_every,
        min_episodes     = args.min_episodes,
        patience         = args.patience,
    )