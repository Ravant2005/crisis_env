"""
evaluate.py — Compare baseline (rule-based) vs trained RL agent.

Runs both agents over the same seeds and prints side-by-side comparison.

Usage:
    python3 evaluate.py                          # default 10 episodes
    python3 evaluate.py --episodes 20
    python3 evaluate.py --checkpoint checkpoints/model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np

from policy_model import PolicyNetwork
from utils import (
    EpisodeSummary,
    observation_to_dict,
    state_to_metrics,
    load_checkpoint,
    set_global_seed,
    run_local_baseline_episode,
)
from models import CrisisAction
from server.environment import CrisisEnvironment


DEFAULT_EPISODES   = 10
DEFAULT_CHECKPOINT = Path("checkpoints/model.pt")


def run_trained_episode(
    env: CrisisEnvironment,
    policy: PolicyNetwork,
) -> EpisodeSummary:
    """
    Run one episode with the trained policy.
    Uses sampling (not greedy) — REINFORCE policies are stochastic by nature,
    and greedy argmax can collapse to a single action type mode.
    """
    observation = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action_dict, _, _ = policy.select_action(observation, greedy=False)
        crisis_action = CrisisAction(**action_dict)
        result = env.step(crisis_action)
        total_reward += float(result.reward)
        done = result.done
        observation = result.observation

    state = env.state()
    task_scores = state_to_metrics(state)
    return EpisodeSummary(
        total_reward=round(total_reward, 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=state.step_count,
    )


def evaluate(
    num_episodes: int = DEFAULT_EPISODES,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Run evaluation comparing baseline vs trained agent."""

    # Load trained policy
    policy = PolicyNetwork(state_dim=229).to(device)
    metadata = load_checkpoint(checkpoint_path, policy, device=device)
    policy.eval()

    print(f"{'='*65}")
    print(f"  Evaluation: Baseline vs Trained Agent")
    print(f"{'='*65}")
    print(f"  Episodes:    {num_episodes}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Model meta:  ep={metadata.get('episode', '?')}, "
          f"best_score={metadata.get('best_score', '?')}")
    print(f"  Device:      {device}")
    print(f"{'='*65}\n")

    baseline_results: List[EpisodeSummary] = []
    trained_results: List[EpisodeSummary] = []

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep

        # Baseline (deterministic)
        baseline_summary = run_local_baseline_episode(ep_seed)
        baseline_results.append(baseline_summary)

        # Trained (stochastic — average over 3 runs for stability)
        trained_scores = []
        trained_rewards = []
        trained_task = {k: [] for k in ["classification", "prediction", "allocation", "coordination", "rescue"]}
        trained_steps = []
        n_runs = 3
        for _ in range(n_runs):
            env = CrisisEnvironment(seed=ep_seed)
            s = run_trained_episode(env, policy)
            trained_scores.append(s.final_score)
            trained_rewards.append(s.total_reward)
            trained_steps.append(s.steps)
            for k in trained_task:
                trained_task[k].append(s.task_scores.get(k, 0.0))

        trained_summary = EpisodeSummary(
            total_reward=round(float(np.mean(trained_rewards)), 4),
            final_score=round(float(np.mean(trained_scores)), 4),
            task_scores={k: round(float(np.mean(v)), 4) for k, v in trained_task.items()},
            steps=int(np.mean(trained_steps)),
        )
        trained_results.append(trained_summary)

        marker = "▲" if trained_summary.final_score > baseline_summary.final_score else "▼"
        print(
            f"Episode {ep:2d} (seed={ep_seed}) | "
            f"Baseline: score={baseline_summary.final_score:.4f} | "
            f"Trained: score={trained_summary.final_score:.4f} "
            f"(std={float(np.std(trained_scores)):.3f}) | {marker}"
        )

    # Aggregate
    def avg_field(results: List[EpisodeSummary], field: str) -> float:
        return sum(getattr(r, field) for r in results) / max(len(results), 1)

    def avg_task(results: List[EpisodeSummary], task: str) -> float:
        return sum(r.task_scores.get(task, 0.0) for r in results) / max(len(results), 1)

    print(f"\n{'='*65}")
    print(f"  RESULTS (averaged over {num_episodes} episodes)")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Trained':>12} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    b_score = avg_field(baseline_results, "final_score")
    t_score = avg_field(trained_results, "final_score")
    print(f"  {'Final Score':<25} {b_score:>12.4f} {t_score:>12.4f} {t_score-b_score:>+10.4f}")

    b_reward = avg_field(baseline_results, "total_reward")
    t_reward = avg_field(trained_results, "total_reward")
    print(f"  {'Total Reward':<25} {b_reward:>12.2f} {t_reward:>12.2f} {t_reward-b_reward:>+10.2f}")

    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for task in ["classification", "prediction", "allocation", "coordination", "rescue"]:
        b = avg_task(baseline_results, task)
        t = avg_task(trained_results, task)
        print(f"  {task.title():<25} {b:>12.4f} {t:>12.4f} {t-b:>+10.4f}")

    print(f"{'='*65}")

    wins = sum(1 for b, t in zip(baseline_results, trained_results) if t.final_score > b.final_score)
    ties = sum(1 for b, t in zip(baseline_results, trained_results) if t.final_score == b.final_score)
    print(f"\n  Trained won: {wins}/{num_episodes} | Tied: {ties}/{num_episodes}")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent vs baseline")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--seed", type=int, default=42)
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

    set_global_seed(args.seed)
    evaluate(
        num_episodes=args.episodes,
        checkpoint_path=Path(args.checkpoint),
        seed=args.seed,
        device=device,
    )
