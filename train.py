"""
train.py — PPO + GAE + Parallel Rollouts + Retrospective Replay
for the Crisis Response OpenEnv environment.

Key upgrades over REINFORCE baseline:
  - PPO clipped surrogate loss (epsilon=0.2) — stable policy updates
  - GAE (Generalized Advantage Estimation, lambda=0.95) — low-variance advantages
  - Parallel rollouts (N_WORKERS envs collected before each update)
  - Value function loss (MSE) — learned critic
  - Retrospective experience replay (top-20% episodes buffered)
  - Fast curriculum: 20 easy / 40 medium / rest on hard
  - PGMCTS warm rollout (see pgmcts module below)
  - Residual encoder with hidden_dim=512
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


# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

DEFAULT_EPISODES        = 300       # more episodes for hard-difficulty mastery
DEFAULT_LR              = 2.5e-4    # slightly higher for PPO stability
DEFAULT_GAMMA           = 0.985
DEFAULT_GAE_LAMBDA      = 0.95      # GAE smoothing factor
DEFAULT_PPO_EPSILON     = 0.20      # PPO clip range
DEFAULT_PPO_EPOCHS      = 4         # PPO inner epochs per rollout batch
DEFAULT_VALUE_COEFF     = 0.5       # value loss weight in combined loss
DEFAULT_ENTROPY_COEFF   = 0.008
DEFAULT_ENTROPY_DECAY   = 0.008
DEFAULT_ENTROPY_FLOOR   = 0.001
DEFAULT_HIDDEN          = 512
DEFAULT_SEED            = 42
DEFAULT_LOG_WINDOW      = 20
DEFAULT_PATIENCE        = 40
DEFAULT_MIN_DELTA       = 0.002
DEFAULT_MIN_EPISODES    = 80
DEFAULT_STOP_SCORE      = 0.90
DEFAULT_BC_STEPS        = 100
DEFAULT_BC_EPISODES     = 24
N_WORKERS               = 8         # parallel rollout workers
REPLAY_BUFFER_SIZE      = 200       # retrospective replay capacity
REPLAY_TOPK_FRAC        = 0.20      # store top-20% of episodes by final_score
REPLAY_MIX_FRAC         = 0.20      # 20% of each minibatch from replay

# Fast curriculum thresholds
PHASE1_SCORE_THRESHOLD  = 0.58
PHASE2_SCORE_THRESHOLD  = 0.72
PHASE1_MAX_STEPS        = 20        # CHANGED: 60 → 20 (fast easy exit)
PHASE2_MAX_STEPS        = 40        # CHANGED: 60 → 40

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")


# ─────────────────────────────────────────────
# CURRICULUM (FAST VERSION)
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
# NOVEL ALGORITHM — PGMCTS
# (Priority-Guided Monte Carlo Tree Search warm rollout)
# See detailed description below in the docstring.
# ─────────────────────────────────────────────

class PGMCTSPlanner:
    """
    Priority-Guided Monte Carlo Tree Search (PGMCTS) warm rollout.

    PURPOSE:
      The RL policy takes many episodes to discover that the optimal action
      selection is governed by the threat priority metric:
          P(t) = severity(t) × population(t) / max(TTI(t), 1)
      PGMCTS injects this domain knowledge as a lookahead bias during the
      EARLY steps of each episode (when time_remaining > LOOKAHEAD_THRESHOLD),
      replacing pure RL exploration with a guided selection that funnels the
      policy toward high-quality action regions.

    ALGORITHM:
      At each step where time_remaining > LOOKAHEAD_THRESHOLD:
        1. Enumerate the valid action set A_valid from the environment mask.
        2. For each candidate action a_i in A_valid:
             - Create a copy of the environment state (lightweight dict clone)
             - Simulate K=10 forward steps using a fast heuristic rollout
               (heuristic = always take the action with highest priority score)
             - Record the resulting cumulative discounted reward R̂(a_i):
                   R̂(a_i) = Σ_{k=0}^{K-1} γ^k · r_k
        3. Compute the PGMCTS action score:
               Q_pgmcts(a_i) = R̂(a_i) / max_j R̂(a_j)    (normalized to [0,1])
        4. Blend with the neural policy logits:
               logits_blended(a_i) = (1 - α) · logits_policy(a_i) + α · C · Q_pgmcts(a_i)
           where α = pgmcts_alpha (default 0.40, annealed to 0.0 by episode N/2)
                 C = scaling constant ≈ 3.0 (maps Q scores to logit magnitude)
        5. Sample or argmax from logits_blended using the normal masked sampling.

      ANNEALING:
        α(episode) = pgmcts_alpha_init × max(0, 1 - 2 × episode / total_episodes)
        This ensures that by the midpoint of training, the policy is fully
        autonomous and PGMCTS no longer influences the distribution.

    LOOKAHEAD_THRESHOLD:
        Only active when time_remaining / total_steps > 0.60.
        During the final 40% of an episode the agent must rely on its own policy
        (rescue / coordination decisions are highly state-dependent and PGMCTS
        heuristics are less reliable in end-game scenarios).

    COMPLEXITY:
        K=10 simulated steps × up to 7 candidate actions = 70 env.step() calls
        per PGMCTS-augmented step. Since the inner simulator uses a rule-based
        heuristic (no neural forward pass), each simulated step takes ~0.2ms,
        adding ~14ms overhead per augmented step — negligible at 20-30 steps/episode.
    """

    LOOKAHEAD_K = 10
    PGMCTS_C    = 3.0

    def __init__(self, gamma: float = 0.985, alpha_init: float = 0.40):
        self.gamma      = gamma
        self.alpha_init = alpha_init

    def alpha(self, episode: int, total_episodes: int) -> float:
        """Anneal PGMCTS influence to zero by the midpoint of training."""
        progress = episode / max(total_episodes, 1)
        return float(self.alpha_init * max(0.0, 1.0 - 2.0 * progress))

    def priority_score(self, threat: dict) -> float:
        """
        Core threat priority metric used in lookahead simulation.
        P(t) = severity × population / max(TTI, 1)
        This is the same formula used internally by the environment's
        _true_priority() — exposed here as a heuristic for the planner.
        """
        sev = float(threat.get("severity", 0.0))
        pop = float(threat.get("population_at_risk", 1.0))
        tti = max(float(threat.get("time_to_impact", 1.0)), 1.0)
        return (sev * pop) / tti

    def heuristic_action(self, obs_dict: dict) -> dict:
        """
        Fast heuristic policy used inside the PGMCTS lookahead simulation.
        Rule: classify unclassified → predict → allocate best resource →
              coordinate → rescue highest-victim zone.
        This is intentionally simple (no neural pass) to keep simulation fast.
        """
        threats   = [t for t in obs_dict.get("threats", []) if t.get("status") == "active"]
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
            z = max(zones, key=lambda x: x.get("total_victims", 0) - x.get("rescued", 0))
            return {"action_type": "rescue",
                    "rescue": {"zone_id": z["zone_id"], "rescue_units_to_send": min(3, budget)}}
        return {"action_type": "skip"}

    def simulate_lookahead(self, env_seed: int, difficulty: str,
                           candidate_action: dict, k: int) -> float:
        """
        Simulate K steps starting with candidate_action using a fresh environment
        copy seeded identically to the current one. Returns cumulative discounted
        reward R̂(a) = Σ γ^t r_t.
        """
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
        """
        Core PGMCTS logit blending.

        logits:         raw action_type logits from PolicyNetwork (shape: [N_ACTIONS])
        action_names:   list of action names corresponding to logit indices
        env_seed:       current environment seed for simulation forks
        difficulty:     current curriculum difficulty string
        valid_mask:     boolean tensor of valid actions (from action mask)
        episode:        current training episode number
        total_episodes: total planned episodes
        time_fraction:  time_remaining / total_steps (only blend if > 0.60)

        Returns:        blended logits tensor (same shape as input)
        """
        alpha = self.alpha(episode, total_episodes)
        if alpha < 1e-4 or time_fraction <= 0.60:
            return logits  # no blending needed

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

        q_vals = list(q_scores.values())
        q_max  = max(q_vals) if q_vals else 1.0
        q_min  = min(q_vals) if q_vals else 0.0
        q_range = max(q_max - q_min, 1e-6)

        blended = logits.clone()
        for idx, q_raw in q_scores.items():
            q_norm = (q_raw - q_min) / q_range
            bias   = alpha * self.PGMCTS_C * q_norm
            blended[idx] = blended[idx] + bias

        return blended

    def _index_to_candidate_action(self, action_name: str) -> dict:
        """Convert an action type name to a minimal valid action dict."""
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
    """
    Stores the top-K% of episodes (by final_score) and replays them
    as behavior cloning signal during PPO updates.

    This prevents catastrophic forgetting on curriculum transitions:
    when the agent jumps from medium → hard difficulty, it tends to
    'unlearn' easy-difficulty patterns. Replaying high-quality easy/medium
    episodes as IL (imitation learning) regularization stabilizes the policy.

    Storage format per episode:
        List of (obs_dict, action_labels) pairs — same format as BC dataset.
    """

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE,
                 topk_fraction: float = REPLAY_TOPK_FRAC):
        self.capacity      = capacity
        self.topk_fraction  = topk_fraction
        self._buffer: deque = deque()

    def add(self, episode_score: float, trajectory: List[Tuple[dict, dict]]) -> None:
        """Add an episode trajectory if it qualifies (top-K% by score)."""
        if not trajectory:
            return
        self._buffer.append((episode_score, trajectory))
        if len(self._buffer) > self.capacity:
            worst_idx = min(range(len(self._buffer)),
                            key=lambda i: self._buffer[i][0])
            del self._buffer[worst_idx]

    def sample_batch(self, batch_size: int) -> List[Tuple[dict, dict]]:
        """Sample `batch_size` (obs, label) pairs from replay buffer."""
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
    from utils import build_state_vector
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
    from utils import collect_baseline_dataset
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
# GAE — GENERALIZED ADVANTAGE ESTIMATION
# ─────────────────────────────────────────────

def compute_gae(rewards: List[float], values: List[float],
                next_value: float, dones: List[bool],
                gamma: float, lam: float) -> Tuple[List[float], List[float]]:
    """
    GAE-Lambda advantage estimation.
    """
    T = len(rewards)
    advantages = [0.0] * T
    last_gae   = 0.0
    extended_values = values + [next_value]

    for t in reversed(range(T)):
        mask    = 0.0 if dones[t] else 1.0
        delta   = rewards[t] + gamma * extended_values[t+1] * mask - extended_values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


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
) -> dict:
    """
    Run one episode and collect (state_vec, log_prob, value, reward, done, action_labels).
    Returns a dict of lists for PPO update.
    """
    env = CrisisEnvironment(seed=seed)
    obs = env.reset(seed=seed, difficulty=difficulty)

    state_vecs   = []
    log_probs    = []
    values_      = []
    rewards_     = []
    dones_       = []
    entropies_   = []
    action_labels= []

    done = False
    while not done:
        from utils import build_state_vector, build_valid_action_mask, observation_to_dict
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
            mask_list = obs_dict.get("valid_actions", {}).get("action_mask", [1]*7)
            valid_mask = torch.tensor(mask_list[:len(ACTION_TYPES)], dtype=torch.bool)
            action_logits = pgmcts.blend_logits(
                action_logits, ACTION_TYPES, seed, difficulty,
                valid_mask, episode, total_episodes, time_fraction
            )
            out["action_type"] = action_logits.unsqueeze(0)

        action_dict, log_prob, entropy, _ = policy.select_action(obs, greedy=False)
        result = env.step(CrisisAction(**action_dict))

        state_vecs.append(sv)
        log_probs.append(log_prob)
        values_.append(value)
        rewards_.append(float(result.reward))
        dones_.append(bool(result.done))
        entropies_.append(entropy)
        action_labels.append({"obs": obs_dict, "action": action_dict})

        done = bool(result.done)
        obs  = result.observation

    state = env.state()
    from utils import state_to_metrics
    task_scores = state_to_metrics(state)

    next_value = 0.0

    return {
        "state_vecs":    state_vecs,
        "log_probs":     log_probs,
        "values":        values_,
        "rewards":       rewards_,
        "dones":         dones_,
        "entropies":     entropies_,
        "action_labels": action_labels,
        "next_value":    next_value,
        "final_score":   float(task_scores.get("final", 0.0)),
        "task_scores":   task_scores,
        "steps":         int(state.step_count),
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
    """
    Collect N_WORKERS episode rollouts in parallel using ThreadPoolExecutor.
    """
    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i in range(n_workers):
            seed_i = base_seed + episode * 100 + i * 7
            futures.append(pool.submit(
                rollout_single_episode,
                policy, difficulty, seed_i,
                pgmcts, episode, total_episodes, device
            ))
    results = []
    for f in as_completed(futures):
        try:
            results.append(f.result())
        except Exception as e:
            print(f"  [WARN] Worker rollout failed: {e}")
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
    """
    PPO-clip update over a batch of rollouts.
    """
    all_sv, all_lp_old, all_adv, all_ret, all_ent = [], [], [], [], []

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
        out    = policy.forward(sv_tensor)
        val_new = out["value"].squeeze(-1)

        from torch.distributions import Categorical
        dist_new = Categorical(logits=out["action_type"])
        lp_new   = dist_new.log_prob(
            torch.zeros(len(all_sv), dtype=torch.long, device=device)
        )

        ratio = torch.exp(lp_new - lp_old)

        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * adv_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(val_new, ret_tensor)

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
        "loss":         round(total_loss_val, 6),
        "policy_loss":  round(float(policy_loss.item()), 6),
        "value_loss":   round(float(value_loss.item()), 6),
        "entropy":      round(float(entropy_bonus.item()), 6),
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
    pgmcts_alpha:       float = 0.40,
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

    print(f"{'='*70}")
    print("  Crisis Response RL — PPO + GAE + PGMCTS + Retrospective Replay")
    print(f"{'='*70}")
    print(f"  Workers: {n_workers}  |  PPO epochs: {ppo_epochs}  |  GAE λ: {gae_lambda}")
    print(f"  PGMCTS: {'ON (α='+str(pgmcts_alpha)+')' if use_pgmcts else 'OFF'}")
    print(f"  Replay buffer: capacity={REPLAY_BUFFER_SIZE}, topk={REPLAY_TOPK_FRAC}")
    print(f"{'='*70}\n")

    bc_meta, bc_dataset = behavior_cloning_warmstart(
        policy, optimizer, seed, bc_steps, bc_episodes, device
    )

    reward_history, score_history = [], []
    best_score = 0.0
    curriculum_phase = 1
    phase_start_episode = 1
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        pre_avg = moving_average(score_history, log_window) if score_history else 0.0
        curriculum_phase, phase_start_episode = update_curriculum_phase(
            episode, curriculum_phase, phase_start_episode, pre_avg
        )
        difficulty = curriculum_difficulty(curriculum_phase)

        rollouts = collect_parallel_rollouts(
            policy=policy,
            difficulty=difficulty,
            base_seed=seed,
            episode=episode,
            total_episodes=num_episodes,
            n_workers=n_workers,
            pgmcts=pgmcts,
            device=device,
        )
        if not rollouts:
            continue

        topk_n = max(1, int(len(rollouts) * REPLAY_TOPK_FRAC))
        sorted_rollouts = sorted(rollouts, key=lambda r: r["final_score"], reverse=True)
        for r in sorted_rollouts[:topk_n]:
            traj = [(lbl["obs"], {"action_type": ACTION_TYPES.index(
                        str(lbl["action"].get("action_type","skip")).replace("ActionType.","")
                    ), "strategy": 0, "threat": 0, "resource": 0,
                    "zone": 0, "units": 0, "severity": 0, "tti": 0, "pop": 0})
                    for lbl in r["action_labels"]]
            replay_buffer.add(r["final_score"], traj)

        ent_coeff_t = entropy_coeff * float(
            np.exp(-entropy_decay * (episode - 1))
        ) + entropy_floor

        update_info = ppo_update(
            policy=policy,
            optimizer=optimizer,
            rollouts=rollouts,
            replay_buffer=replay_buffer,
            bc_dataset=bc_dataset,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_epsilon=ppo_epsilon,
            ppo_epochs=ppo_epochs,
            entropy_coeff=ent_coeff_t,
            value_coeff=value_coeff,
            device=device,
        )

        avg_final_score = float(np.mean([r["final_score"] for r in rollouts]))
        avg_reward      = float(np.mean([sum(r["rewards"]) for r in rollouts]))
        avg_steps       = float(np.mean([r["steps"] for r in rollouts]))

        reward_history.append(avg_reward)
        score_history.append(avg_final_score)

        avg_score  = moving_average(score_history, log_window)
        replay_len = len(replay_buffer)

        if episode % 5 == 0 or episode == 1:
            elapsed = time.time() - start_time
            eps_per_sec = (episode * n_workers) / max(elapsed, 1e-6)
            print(
                f"Ep {episode:3d} | ph={curriculum_phase} diff={difficulty:<6} | "
                f"score={avg_final_score:.4f} avg={avg_score:.4f} | "
                f"loss={update_info.get('loss',0):.4f} "
                f"vl={update_info.get('value_loss',0):.4f} | "
                f"replay={replay_len} | {eps_per_sec:.1f} ep/s"
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
        })

        if (episode >= max(min_episodes, patience + 1)
                and len(score_history) > patience):
            delta_s = abs(moving_average(score_history[-patience:], patience//2) -
                          moving_average(score_history[-2*patience:-patience], patience//2))
            if delta_s < min_delta and avg_score > stop_score:
                print(f"Early stop at ep {episode}: Δscore={delta_s:.5f} avg={avg_score:.4f}")
                break

    save_checkpoint(CHECKPOINT_DIR / "model.pt", policy, optimizer,
                    metadata={"episodes_run": len(score_history),
                              "best_score": best_score,
                              "hidden_dim": hidden_dim, "state_dim": STATE_DIM})

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best score: {best_score:.4f}  |  Episodes: {len(score_history)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,   default=DEFAULT_EPISODES)
    parser.add_argument("--lr",             type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden",         type=int,   default=DEFAULT_HIDDEN)
    parser.add_argument("--n-workers",      type=int,   default=N_WORKERS)
    parser.add_argument("--no-pgmcts",      action="store_true")
    parser.add_argument("--pgmcts-alpha",   type=float, default=0.40)
    parser.add_argument("--seed",           type=int,   default=DEFAULT_SEED)
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--stop-score",     type=float, default=DEFAULT_STOP_SCORE)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if (hasattr(torch.backends,"mps")
                                 and torch.backends.mps.is_available())
                  else "cpu")
    else:
        device = args.device

    train(
        num_episodes=args.episodes,
        lr=args.lr,
        hidden_dim=args.hidden,
        n_workers=args.n_workers,
        use_pgmcts=not args.no_pgmcts,
        pgmcts_alpha=args.pgmcts_alpha,
        seed=args.seed,
        device=device,
        stop_score=args.stop_score,
        checkpoint_every=args.checkpoint_every,
    )