"""
train.py — Fast hierarchical RL training for the Crisis Response environment.

Key features:
- Hierarchical policy (strategy -> action)
- Action masking for efficient exploration
- Progression-gated curriculum (easy -> medium -> hard)
- Optional behavior cloning warm-start
- Adaptive early stopping with entropy gate
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models import CrisisAction
from policy_model import PolicyNetwork
from server.environment import CrisisEnvironment
from utils import (
    ACTION_TYPES,
    STATE_DIM,
    STRATEGY_TYPES,
    EpisodeSummary,
    TrainingLogger,
    build_state_vector,
    collect_baseline_dataset,
    compute_discounted_returns,
    load_checkpoint,
    moving_average,
    observation_to_dict,
    save_checkpoint,
    set_global_seed,
    state_to_metrics,
)


# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────

DEFAULT_EPISODES = 140
DEFAULT_LR = 1.8e-4
DEFAULT_GAMMA = 0.985
DEFAULT_HIDDEN = 256
DEFAULT_SEED = 42
DEFAULT_ENTROPY = 0.006
DEFAULT_ENTROPY_DECAY = 0.012
DEFAULT_ENTROPY_MIN = 0.90
DEFAULT_ENTROPY_FLOOR = 0.001
DEFAULT_LOG_WINDOW = 20
DEFAULT_PATIENCE = 32
DEFAULT_MIN_DELTA = 0.002
DEFAULT_MIN_EPISODES = 100
DEFAULT_STOP_SCORE_THRESHOLD = 0.85
DEFAULT_BC_STEPS = 80
DEFAULT_BC_EPISODES = 16
DEFAULT_BALANCE_LOSS = 0.08
DEFAULT_BALANCE_BETA = 0.03

PHASE1_SCORE_THRESHOLD = 0.62
PHASE2_SCORE_THRESHOLD = 0.74
PHASE1_MAX_STEPS = 60
PHASE2_MAX_STEPS = 60

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")


# ─────────────────────────────────────────────
# CURRICULUM
# ─────────────────────────────────────────────


def curriculum_difficulty(phase: int) -> str:
    if phase <= 1:
        return "easy"
    if phase == 2:
        return "medium"
    return "hard"


def update_curriculum_phase(
    episode: int,
    current_phase: int,
    phase_start_episode: int,
    avg_score: float,
    max_phase_steps_1: int = PHASE1_MAX_STEPS,
    max_phase_steps_2: int = PHASE2_MAX_STEPS,
) -> Tuple[int, int]:
    """
    Forced curriculum:
      Phase 1: episodes 1..50 (easy)
      Phase 2: medium
      Phase 3: hard

    Hybrid advancement:
      advance if avg_score > threshold OR phase exceeds max_phase_steps.
    """
    phase_age = episode - phase_start_episode + 1

    if current_phase == 1:
        if avg_score >= PHASE1_SCORE_THRESHOLD or phase_age >= max_phase_steps_1:
            return 2, episode
        return 1, phase_start_episode

    if current_phase == 2:
        if avg_score >= PHASE2_SCORE_THRESHOLD or phase_age >= max_phase_steps_2:
            return 3, episode
        return 2, phase_start_episode

    return 3, phase_start_episode


# ─────────────────────────────────────────────
# BEHAVIOR CLONING WARM-START
# ─────────────────────────────────────────────


def _bc_loss_for_sample(policy: PolicyNetwork, obs: Dict, labels: Dict[str, int], device: str) -> torch.Tensor:
    state_vec = build_state_vector(obs)
    x = torch.tensor(state_vec, dtype=torch.float32, device=device)
    logits = {k: v.squeeze(0) for k, v in policy.forward(x).items()}

    loss = torch.tensor(0.0, device=device)

    loss = loss + F.cross_entropy(logits["strategy"].unsqueeze(0), torch.tensor([labels["strategy"]], device=device))
    loss = loss + F.cross_entropy(logits["action_type"].unsqueeze(0), torch.tensor([labels["action_type"]], device=device))

    action_name = ACTION_TYPES[labels["action_type"]]

    if action_name in {"classify", "predict", "allocate", "delay"}:
        loss = loss + F.cross_entropy(logits["threat"].unsqueeze(0), torch.tensor([labels["threat"]], device=device))

    if action_name == "allocate":
        loss = loss + F.cross_entropy(logits["resource"].unsqueeze(0), torch.tensor([labels["resource"]], device=device))

    if action_name == "rescue":
        loss = loss + F.cross_entropy(logits["zone"].unsqueeze(0), torch.tensor([labels["zone"]], device=device))
        loss = loss + F.cross_entropy(logits["units"].unsqueeze(0), torch.tensor([labels["units"]], device=device))

    if action_name == "delay":
        loss = loss + F.cross_entropy(logits["units"].unsqueeze(0), torch.tensor([labels["units"]], device=device))

    if action_name == "classify":
        loss = loss + F.cross_entropy(logits["severity"].unsqueeze(0), torch.tensor([labels["severity"]], device=device))

    if action_name == "predict":
        loss = loss + F.cross_entropy(logits["tti"].unsqueeze(0), torch.tensor([labels["tti"]], device=device))
        loss = loss + F.cross_entropy(logits["pop"].unsqueeze(0), torch.tensor([labels["pop"]], device=device))

    return loss


def behavior_cloning_warmstart(
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    seed: int,
    bc_steps: int,
    bc_episodes: int,
    device: str,
) -> Tuple[Dict[str, float], List[Tuple[Dict, Dict]]]:
    if bc_steps <= 0 or bc_episodes <= 0:
        return {"enabled": False, "steps": 0, "final_loss": 0.0}, []

    datasets: List[Tuple[Dict, Dict]] = []
    per_diff = max(1, bc_episodes // 3)
    datasets.extend(collect_baseline_dataset(per_diff, seed + 100, difficulty="easy"))
    datasets.extend(collect_baseline_dataset(per_diff, seed + 200, difficulty="medium"))
    datasets.extend(collect_baseline_dataset(max(1, bc_episodes - 2 * per_diff), seed + 300, difficulty="hard"))

    if not datasets:
        return {"enabled": False, "steps": 0, "final_loss": 0.0}, []

    rng = random.Random(seed + 404)
    batch_size = 32
    final_loss = 0.0

    policy.train()
    for step in range(1, bc_steps + 1):
        batch = [datasets[rng.randrange(len(datasets))] for _ in range(batch_size)]

        loss = torch.tensor(0.0, device=device)
        for obs, labels in batch:
            loss = loss + _bc_loss_for_sample(policy, obs, labels, device)
        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        final_loss = float(loss.item())

    return {"enabled": True, "steps": bc_steps, "final_loss": round(final_loss, 6), "samples": len(datasets)}, datasets


def estimate_baseline_action_distribution_from_bc(
    bc_dataset: List[Tuple[Dict, Dict]],
) -> Dict[str, float]:
    """
    Estimate p*(a) from baseline trajectories collected for behavior cloning.
    """
    counts = {a: 0.0 for a in ["classify", "predict", "allocate", "coordinate", "rescue"]}
    if not bc_dataset:
        uniform = 1.0 / len(counts)
        return {k: uniform for k in counts}

    for _, labels in bc_dataset:
        idx = int(labels.get("action_type", ACTION_TYPES.index("skip")))
        action_name = ACTION_TYPES[idx] if 0 <= idx < len(ACTION_TYPES) else "skip"
        if action_name in counts:
            counts[action_name] += 1.0

    total = sum(counts.values())
    if total <= 0:
        uniform = 1.0 / len(counts)
        return {k: uniform for k in counts}
    return {k: counts[k] / total for k in counts}


# ─────────────────────────────────────────────
# ROLLOUT
# ─────────────────────────────────────────────


def rollout_episode(
    env: CrisisEnvironment,
    policy: PolicyNetwork,
    difficulty: str,
    episode_seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], EpisodeSummary, Dict[str, int]]:
    observation = env.reset(seed=episode_seed, difficulty=difficulty)

    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    action_counts: Dict[str, int] = {k: 0 for k in ["classify", "predict", "allocate", "coordinate", "rescue"]}

    done = False

    while not done:
        action_dict, log_prob, entropy, _ = policy.select_action(observation, greedy=False)
        result = env.step(CrisisAction(**action_dict))

        action_type = str(action_dict.get("action_type", "skip"))
        if action_type in action_counts:
            action_counts[action_type] += 1

        log_probs.append(log_prob)
        entropies.append(entropy)
        rewards.append(float(result.reward))

        done = bool(result.done)
        observation = result.observation

    state = env.state()
    task_scores = state_to_metrics(state)
    summary = EpisodeSummary(
        total_reward=round(float(sum(rewards)), 4),
        final_score=round(float(task_scores["final"]), 4),
        task_scores=task_scores,
        steps=int(state.step_count),
    )
    return log_probs, entropies, rewards, summary, action_counts


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────


def train(
    num_episodes: int = DEFAULT_EPISODES,
    lr: float = DEFAULT_LR,
    gamma: float = DEFAULT_GAMMA,
    hidden_dim: int = DEFAULT_HIDDEN,
    entropy_coeff: float = DEFAULT_ENTROPY,
    seed: int = DEFAULT_SEED,
    checkpoint_every: int = 50,
    log_window: int = DEFAULT_LOG_WINDOW,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    stop_score_threshold: float = DEFAULT_STOP_SCORE_THRESHOLD,
    entropy_min: float = DEFAULT_ENTROPY_MIN,
    entropy_decay: float = DEFAULT_ENTROPY_DECAY,
    entropy_floor: float = DEFAULT_ENTROPY_FLOOR,
    bc_steps: int = DEFAULT_BC_STEPS,
    bc_episodes: int = DEFAULT_BC_EPISODES,
    balance_loss_coeff: float = DEFAULT_BALANCE_LOSS,
    balance_beta: float = DEFAULT_BALANCE_BETA,
    device: str = "cpu",
) -> None:
    set_global_seed(seed)

    policy = PolicyNetwork(state_dim=STATE_DIM, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    logger = TrainingLogger(LOG_DIR / "training.jsonl", clear=True)

    print(f"{'='*74}")
    print("  Crisis Response RL Training (Hierarchical + Masked + Curriculum)")
    print(f"{'='*74}")
    print(f"  Episodes:            {num_episodes}")
    print(f"  Learning rate:       {lr}")
    print(f"  Gamma:               {gamma}")
    print(f"  Entropy coeff:       {entropy_coeff} (decay={entropy_decay})")
    print(f"  Hidden dim:          {hidden_dim}")
    print(f"  Min episodes:        {min_episodes}")
    print(f"  Stop score floor:    {stop_score_threshold}")
    print(f"  Early-stop K:        {patience}")
    print(f"  Early-stop eps:      {min_delta}")
    print(f"  Entropy threshold:   {entropy_min}")
    print(f"  Entropy floor:       {entropy_floor}")
    print(f"  Balance loss coeff:  {balance_loss_coeff} (inactive)")
    print(f"  Balance beta:        {balance_beta} (inactive)")
    print(f"  Device:              {device}")
    print(f"{'='*74}")

    bc_meta, bc_dataset = behavior_cloning_warmstart(policy, optimizer, seed, bc_steps, bc_episodes, device)
    baseline_action_dist = estimate_baseline_action_distribution_from_bc(bc_dataset)
    if bc_meta.get("enabled"):
        print(
            f"  BC warm-start:       enabled (steps={bc_meta['steps']}, samples={bc_meta['samples']}, "
            f"final_loss={bc_meta['final_loss']})"
        )
    else:
        print("  BC warm-start:       disabled")
    print(f"{'='*74}\n")

    reward_history: List[float] = []
    score_history: List[float] = []
    entropy_history: List[float] = []
    avg_score_history: List[float] = []
    avg_entropy_history: List[float] = []

    best_score = 0.0
    curriculum_phase = 1
    phase_start_episode = 1

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        pre_avg_score = moving_average(score_history, log_window) if score_history else 0.0
        curriculum_phase, phase_start_episode = update_curriculum_phase(
            episode=episode,
            current_phase=curriculum_phase,
            phase_start_episode=phase_start_episode,
            avg_score=pre_avg_score,
        )
        difficulty = curriculum_difficulty(curriculum_phase)
        env = CrisisEnvironment(seed=seed + episode * 11)
        env.set_baseline_action_distribution(baseline_action_dist)

        log_probs, entropies, rewards, summary, action_counts = rollout_episode(
            env=env,
            policy=policy,
            difficulty=difficulty,
            episode_seed=seed + episode * 11,
        )

        if len(log_probs) >= 2:
            returns = compute_discounted_returns(rewards, gamma)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

            # Advantage normalization (variance reduction, stable gradients).
            value_baseline = returns_tensor.mean()
            advantages = returns_tensor - value_baseline
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).mean()
            entropy_bonus = torch.stack(entropies).mean()
            entropy_coeff_t = entropy_coeff * float(np.exp(-entropy_decay * (episode - 1))) + entropy_floor
            loss = policy_loss - entropy_coeff_t * entropy_bonus

            # Keep policy close to strong demonstrated behavior to avoid collapse.
            if bc_dataset:
                il_batch_size = 8
                il_loss = torch.tensor(0.0, device=device)
                for _ in range(il_batch_size):
                    obs_i, labels_i = random.choice(bc_dataset)
                    il_loss = il_loss + _bc_loss_for_sample(policy, obs_i, labels_i, device)
                il_loss = il_loss / il_batch_size
                loss = loss + 0.04 * il_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = float(loss.item())
            entropy_value = float(entropy_bonus.item())
            entropy_coeff_value = float(entropy_coeff_t)
            balance_kl_value = 0.0
        else:
            loss_value = 0.0
            entropy_value = float(torch.stack(entropies).mean().item()) if entropies else 0.0
            entropy_coeff_value = float(entropy_coeff * float(np.exp(-entropy_decay * (episode - 1))) + entropy_floor)
            balance_kl_value = 0.0

        reward_history.append(summary.total_reward)
        score_history.append(summary.final_score)
        entropy_history.append(entropy_value)

        avg_reward = moving_average(reward_history, log_window)
        avg_score = moving_average(score_history, log_window)
        avg_entropy = moving_average(entropy_history, log_window)
        avg_score_history.append(avg_score)
        avg_entropy_history.append(avg_entropy)

        logger.write(
            {
                "episode": episode,
                "phase": curriculum_phase,
                "difficulty": difficulty,
                "total_reward": summary.total_reward,
                "final_score": summary.final_score,
                "avg_reward": round(avg_reward, 4),
                "avg_score": round(avg_score, 4),
                "avg_entropy": round(avg_entropy, 4),
                "steps": summary.steps,
                "task_scores": summary.task_scores,
                "loss": round(loss_value, 6),
                "entropy": round(entropy_value, 6),
                "entropy_coeff_t": round(entropy_coeff_value, 6),
                "balance_kl": round(balance_kl_value, 6),
                "action_counts": action_counts,
                "baseline_action_dist": baseline_action_dist,
            }
        )

        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            eps_per_sec = episode / max(elapsed, 1e-6)
            print(
                f"Episode {episode:3d} | phase={curriculum_phase} | diff={difficulty:<6} | "
                f"score={summary.final_score:.4f} | avg={avg_score:.4f} | "
                f"H={avg_entropy:.3f} | reward={summary.total_reward:7.3f} | "
                f"steps={summary.steps:2d} | {eps_per_sec:.1f} ep/s"
            )

        if summary.final_score > best_score:
            best_score = summary.final_score
            save_checkpoint(
                CHECKPOINT_DIR / "best_model.pt",
                policy,
                optimizer,
                metadata={
                    "episode": episode,
                    "best_score": best_score,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "state_dim": STATE_DIM,
                    "strategies": STRATEGY_TYPES,
                    "actions": ACTION_TYPES,
                    "curriculum_phase": curriculum_phase,
                    "entropy_decay": entropy_decay,
                    "entropy_floor": entropy_floor,
                    "stop_score_threshold": stop_score_threshold,
                    "baseline_action_dist": baseline_action_dist,
                },
            )

        if episode % checkpoint_every == 0:
            save_checkpoint(
                CHECKPOINT_DIR / f"checkpoint_ep{episode}.pt",
                policy,
                optimizer,
                metadata={
                    "episode": episode,
                    "final_score": summary.final_score,
                    "avg_score": avg_score,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "state_dim": STATE_DIM,
                    "curriculum_phase": curriculum_phase,
                    "baseline_action_dist": baseline_action_dist,
                },
            )

        # Adaptive early stopping:
        # Stop only if score improvement is tiny AND policy entropy is low.
        if episode >= max(min_episodes, patience + 1) and len(avg_score_history) > patience:
            delta_s = abs(avg_score_history[-1] - avg_score_history[-1 - patience])
            low_improvement = delta_s < min_delta
            low_entropy = avg_entropy_history[-1] < entropy_min
            high_enough_score = avg_score_history[-1] > stop_score_threshold
            if low_improvement and low_entropy and high_enough_score:
                print(
                    f"Adaptive early stop at episode {episode}: "
                    f"delta_s={delta_s:.6f} < {min_delta}, "
                    f"avg_entropy={avg_entropy_history[-1]:.4f} < {entropy_min}, "
                    f"avg_score={avg_score_history[-1]:.4f} > {stop_score_threshold}."
                )
                break

    total_episodes_run = len(score_history)

    save_checkpoint(
        CHECKPOINT_DIR / "model.pt",
        policy,
        optimizer,
        metadata={
            "episode": total_episodes_run,
            "best_score": best_score,
            "final_score": score_history[-1] if score_history else 0.0,
            "avg_score": moving_average(score_history, log_window),
            "seed": seed,
            "hidden_dim": hidden_dim,
            "state_dim": STATE_DIM,
            "strategies": STRATEGY_TYPES,
            "actions": ACTION_TYPES,
            "bc": bc_meta,
            "min_episodes": min_episodes,
            "entropy_min": entropy_min,
            "entropy_decay": entropy_decay,
            "entropy_floor": entropy_floor,
            "stop_score_threshold": stop_score_threshold,
            "curriculum_phase": curriculum_phase,
            "baseline_action_dist": baseline_action_dist,
        },
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*74}")
    print("  Training complete")
    print(f"  Episodes run:        {total_episodes_run}")
    print(f"  Best final score:    {best_score:.4f}")
    print(f"  Last avg score:      {moving_average(score_history, log_window):.4f}")
    print(f"  Total time:          {elapsed:.1f}s")
    print(f"  Logs:                {LOG_DIR / 'training.jsonl'}")
    print(f"  Checkpoints:         {CHECKPOINT_DIR}/")
    print(f"{'='*74}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for Crisis Response")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--entropy", type=float, default=DEFAULT_ENTROPY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--min-episodes", type=int, default=DEFAULT_MIN_EPISODES)
    parser.add_argument("--stop-score-threshold", type=float, default=DEFAULT_STOP_SCORE_THRESHOLD)
    parser.add_argument("--entropy-min", type=float, default=DEFAULT_ENTROPY_MIN)
    parser.add_argument("--entropy-decay", type=float, default=DEFAULT_ENTROPY_DECAY)
    parser.add_argument("--entropy-floor", type=float, default=DEFAULT_ENTROPY_FLOOR)
    parser.add_argument("--bc-steps", type=int, default=DEFAULT_BC_STEPS)
    parser.add_argument("--bc-episodes", type=int, default=DEFAULT_BC_EPISODES)
    parser.add_argument("--balance-loss", type=float, default=DEFAULT_BALANCE_LOSS)
    parser.add_argument("--balance-beta", type=float, default=DEFAULT_BALANCE_BETA)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    train(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden,
        entropy_coeff=args.entropy,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        patience=args.patience,
        min_delta=args.min_delta,
        min_episodes=args.min_episodes,
        stop_score_threshold=args.stop_score_threshold,
        entropy_min=args.entropy_min,
        entropy_decay=args.entropy_decay,
        entropy_floor=args.entropy_floor,
        bc_steps=args.bc_steps,
        bc_episodes=args.bc_episodes,
        balance_loss_coeff=args.balance_loss,
        balance_beta=args.balance_beta,
        device=device,
    )
