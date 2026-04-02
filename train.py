"""
train.py — True Reinforcement Learning training loop for the Crisis Response Environment.

Trains a multi-head policy network directly against CrisisEnvironment.
The agent outputs action components (type, threat, resource, zone, params)
without any pre-computed candidates or heuristic optimal solutions.

Key design:
- Reward = average of 5 grader scores (aligned with final_score)
- Entropy bonus across all action component heads
- Reward clipping, return normalization, gradient clipping
- Clean logging (file cleared on each run)

Usage:
    python3 train.py                           # 500 episodes, lr=1e-4
    python3 train.py --episodes 1000           # longer training
    python3 train.py --lr 5e-5 --entropy 0.03 # tuned hyperparameters
    python3 train.py --seed 123                # different seed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from policy_model import PolicyNetwork
from utils import (
    EpisodeSummary,
    TrainingLogger,
    build_state_vector,
    compute_discounted_returns,
    decode_action,
    moving_average,
    observation_to_dict,
    save_checkpoint,
    load_checkpoint,
    set_global_seed,
    state_to_metrics,
)
from models import CrisisAction
from server.environment import CrisisEnvironment


# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────

DEFAULT_EPISODES   = 500
DEFAULT_LR         = 1e-4        # low LR for stability
DEFAULT_GAMMA      = 0.99
DEFAULT_HIDDEN     = 256
DEFAULT_SEED       = 42
DEFAULT_ENTROPY    = 0.05        # higher entropy to prevent mode collapse
DEFAULT_LOG_WINDOW = 20
CHECKPOINT_DIR     = Path("checkpoints")
LOG_DIR            = Path("logs")

MAX_REWARD_CLIP = 1.0  # clip step rewards to [-1, 1]


# ─────────────────────────────────────────────
# REWARD COMPUTATION (grader-aligned)
# ─────────────────────────────────────────────

def compute_step_reward(env: CrisisEnvironment) -> float:
    """
    Reward = average of current grader scores + small step bonus.
    This aligns the step reward with the final evaluation score.

    The 5 graders each output 0.0–1.0, so the average is 0.0–1.0.
    A small step bonus (+0.002 per step) encourages the agent to
    take actions that make progress rather than doing nothing.
    """
    scores = env.task_scores()
    avg_score = sum(scores.values()) / 5.0
    step_bonus = 0.002  # tiny incentive to take actions
    return float(avg_score + step_bonus)


# ─────────────────────────────────────────────
# SINGLE EPISODE ROLLOUT
# ─────────────────────────────────────────────

def rollout_episode(
    env: CrisisEnvironment,
    policy: PolicyNetwork,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]], List[float], EpisodeSummary]:
    """
    Run one full episode with the policy network.
    Returns (log_probs, all_logits, rewards, summary).

    The policy DIRECTLY outputs action components — no heuristic
    candidate selection or pre-computed optimal actions.
    """
    observation = env.reset()

    log_probs: List[torch.Tensor] = []
    all_logits: List[Dict[str, torch.Tensor]] = []
    rewards: List[float] = []
    done = False
    steps = 0

    while not done:
        action_dict, log_prob, logits = policy.select_action(observation, greedy=False)

        # Move to device
        log_prob = log_prob.to(device)
        logits = {k: v.to(device) for k, v in logits.items()}

        # Execute action
        crisis_action = CrisisAction(**action_dict)
        result = env.step(crisis_action)

        # Grader-aligned reward
        if result.done:
            # Terminal: use final score directly
            state = env.state()
            reward = float(state.final_score)
        else:
            reward = compute_step_reward(env)

        # Clip reward for stability
        reward = max(min(reward, MAX_REWARD_CLIP), -MAX_REWARD_CLIP)

        log_probs.append(log_prob)
        all_logits.append(logits)
        rewards.append(reward)
        done = result.done
        observation = result.observation
        steps += 1

    state = env.state()
    task_scores = state_to_metrics(state)
    summary = EpisodeSummary(
        total_reward=round(sum(rewards), 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=state.step_count,
    )

    return log_probs, all_logits, rewards, summary


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
    device: str = "cpu",
) -> None:
    """Main REINFORCE training loop."""

    set_global_seed(seed)

    # Policy
    policy = PolicyNetwork(
        state_dim=229,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Clean logging
    log_path = LOG_DIR / "training.jsonl"
    logger = TrainingLogger(log_path, clear=True)

    print(f"{'='*65}")
    print(f"  Crisis Response — True RL Training (REINFORCE)")
    print(f"{'='*65}")
    print(f"  Episodes:       {num_episodes}")
    print(f"  Learning rate:  {lr}")
    print(f"  Gamma:          {gamma}")
    print(f"  Hidden dim:     {hidden_dim}")
    print(f"  Entropy coeff:  {entropy_coeff}")
    print(f"  Reward clip:    [{-MAX_REWARD_CLIP}, {MAX_REWARD_CLIP}]")
    print(f"  Seed:           {seed}")
    print(f"  Device:         {device}")
    print(f"{'='*65}\n")

    reward_history: List[float] = []
    score_history: List[float] = []
    best_score = 0.0
    avg_reward = 0.0
    avg_score = 0.0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        env = CrisisEnvironment(seed=seed + episode)

        log_probs, all_logits, rewards, summary = rollout_episode(env, policy, device)

        if len(log_probs) < 2:
            # Too short for meaningful learning — skip update but log
            reward_history.append(summary.total_reward)
            score_history.append(summary.final_score)
            avg_reward = moving_average(reward_history, log_window)
            avg_score = moving_average(score_history, log_window)
            logger.write({
                "episode": episode, "total_reward": summary.total_reward,
                "final_score": summary.final_score, "steps": summary.steps,
                "avg_reward": round(avg_reward, 4), "avg_score": round(avg_score, 4),
                "skipped": True,
            })
            continue

        # Discounted returns
        returns = compute_discounted_returns(rewards, gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalise returns (only for episodes with enough steps)
        if returns_tensor.numel() > 2:
            std = returns_tensor.std()
            if std > 1e-8:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (std + 1e-8)

        # REINFORCE policy gradient
        policy_loss = torch.stack([
            -lp * ret for lp, ret in zip(log_probs, returns_tensor)
        ]).sum()

        # Entropy bonus (across all component heads)
        entropy_loss = torch.tensor(0.0, device=device)
        for logits_dict in all_logits:
            entropy_loss -= entropy_coeff * policy.entropy(logits_dict)

        loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Track
        reward_history.append(summary.total_reward)
        score_history.append(summary.final_score)
        avg_reward = moving_average(reward_history, log_window)
        avg_score = moving_average(score_history, log_window)

        logger.write({
            "episode": episode, "total_reward": summary.total_reward,
            "final_score": summary.final_score,
            "task_scores": summary.task_scores, "steps": summary.steps,
            "avg_reward": round(avg_reward, 4), "avg_score": round(avg_score, 4),
            "loss": round(loss.item(), 6),
        })

        # Print
        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            eps_per_sec = episode / max(elapsed, 1e-6)
            print(
                f"Episode {episode:4d} | "
                f"reward: {summary.total_reward:6.3f} | "
                f"score: {summary.final_score:.4f} | "
                f"avg_score: {avg_score:.4f} | "
                f"steps: {summary.steps:2d} | "
                f"{eps_per_sec:.1f} ep/s"
            )

        # Best model
        if summary.final_score > best_score:
            best_score = summary.final_score
            save_checkpoint(
                CHECKPOINT_DIR / "best_model.pt", policy, optimizer,
                metadata={"episode": episode, "best_score": best_score,
                           "seed": seed, "hidden_dim": hidden_dim},
            )

        # Periodic checkpoint
        if episode % checkpoint_every == 0:
            save_checkpoint(
                CHECKPOINT_DIR / f"checkpoint_ep{episode}.pt", policy, optimizer,
                metadata={"episode": episode, "final_score": summary.final_score,
                           "avg_score": avg_score, "seed": seed, "hidden_dim": hidden_dim},
            )

    # Final save
    save_checkpoint(
        CHECKPOINT_DIR / "model.pt", policy, optimizer,
        metadata={"episode": num_episodes, "best_score": best_score,
                   "final_score": score_history[-1] if score_history else 0.0,
                   "seed": seed, "hidden_dim": hidden_dim},
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"  Training complete!")
    print(f"  Best final score:    {best_score:.4f}")
    print(f"  Last avg score ({log_window}): {avg_score:.4f}")
    print(f"  Total time:          {elapsed:.1f}s")
    print(f"  Logs:                {log_path}")
    print(f"  Checkpoints:         {CHECKPOINT_DIR}/")
    print(f"{'='*65}")


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
        num_episodes=args.episodes, lr=args.lr, gamma=args.gamma,
        hidden_dim=args.hidden, entropy_coeff=args.entropy, seed=args.seed,
        checkpoint_every=args.checkpoint_every, device=device,
    )
