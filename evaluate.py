"""
evaluate.py — Compare weakened baseline vs trained hierarchical RL agent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

from models import CrisisAction
from policy_model import PolicyNetwork
from server.environment import CrisisEnvironment
from utils import (
    ACTION_TYPES,
    STATE_DIM,
    STRATEGY_TYPES,
    EpisodeSummary,
    load_checkpoint,
    run_local_baseline_episode,
    set_global_seed,
    state_to_metrics,
)


DEFAULT_EPISODES = 10
DEFAULT_CHECKPOINT = Path("checkpoints/model.pt")


def run_trained_episode(env: CrisisEnvironment, policy: PolicyNetwork) -> EpisodeSummary:
    observation = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action_dict, _, _, _ = policy.select_action(observation, greedy=True)
        result = env.step(CrisisAction(**action_dict))
        total_reward += float(result.reward)
        done = bool(result.done)
        observation = result.observation

    state = env.state()
    task_scores = state_to_metrics(state)
    return EpisodeSummary(
        total_reward=round(total_reward, 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=int(state.step_count),
    )


def evaluate(
    num_episodes: int = DEFAULT_EPISODES,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    seed: int = 42,
    device: str = "cpu",
    hidden_dim: int = 256,
) -> None:
    # First, quickly peek at the checkpoint to infer the correct hidden_dim
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", {})
        if "input_proj.weight" in sd:
            hidden_dim = sd["input_proj.weight"].shape[0]
    except Exception:
        pass
        
    policy = PolicyNetwork(state_dim=STATE_DIM, hidden_dim=hidden_dim).to(device)
    metadata = load_checkpoint(checkpoint_path, policy, device=device)
    policy.eval()

    print(f"{'='*70}")
    print("  Evaluation: Baseline vs Trained")
    print(f"{'='*70}")
    print(f"  Episodes:    {num_episodes}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(
        f"  Model meta:  ep={metadata.get('episode', '?')}, "
        f"best_score={metadata.get('best_score', '?')}"
    )
    print(f"  Actions:     {metadata.get('actions', ACTION_TYPES)}")
    print(f"  Strategies:  {metadata.get('strategies', STRATEGY_TYPES)}")
    print(f"  Device:      {device}")
    print(f"{'='*70}\n")

    baseline_results: List[EpisodeSummary] = []
    trained_results: List[EpisodeSummary] = []

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep

        baseline = run_local_baseline_episode(ep_seed, difficulty="medium")
        baseline_results.append(baseline)

        trained_scores = []
        trained_rewards = []
        trained_steps = []
        trained_task = {k: [] for k in ["classification", "prediction", "allocation", "coordination", "rescue"]}

        for _ in range(3):
            env = CrisisEnvironment(seed=ep_seed)
            s = run_trained_episode(env, policy)
            trained_scores.append(s.final_score)
            trained_rewards.append(s.total_reward)
            trained_steps.append(s.steps)
            for k in trained_task:
                trained_task[k].append(s.task_scores.get(k, 0.0))

        trained = EpisodeSummary(
            total_reward=round(float(np.mean(trained_rewards)), 4),
            final_score=round(float(np.mean(trained_scores)), 4),
            task_scores={k: round(float(np.mean(v)), 4) for k, v in trained_task.items()},
            steps=int(np.mean(trained_steps)),
        )
        trained_results.append(trained)

        marker = "▲" if trained.final_score > baseline.final_score else "▼"
        print(
            f"Episode {ep:2d} (seed={ep_seed}) | "
            f"Baseline={baseline.final_score:.4f} | Trained={trained.final_score:.4f} "
            f"(std={float(np.std(trained_scores)):.3f}) | {marker}"
        )

    def avg_field(results: List[EpisodeSummary], field: str) -> float:
        return sum(getattr(r, field) for r in results) / max(1, len(results))

    def avg_task(results: List[EpisodeSummary], task: str) -> float:
        return sum(r.task_scores.get(task, 0.0) for r in results) / max(1, len(results))

    b_score = avg_field(baseline_results, "final_score")
    t_score = avg_field(trained_results, "final_score")
    b_reward = avg_field(baseline_results, "total_reward")
    t_reward = avg_field(trained_results, "total_reward")

    print(f"\n{'='*70}")
    print("  Aggregate")
    print(f"{'='*70}")
    print(f"  {'Metric':<22} {'Baseline':>12} {'Trained':>12} {'Delta':>10}")
    print(f"  {'Final Score':<22} {b_score:>12.4f} {t_score:>12.4f} {t_score-b_score:>+10.4f}")
    print(f"  {'Total Reward':<22} {b_reward:>12.4f} {t_reward:>12.4f} {t_reward-b_reward:>+10.4f}")

    for task in ["classification", "prediction", "allocation", "coordination", "rescue"]:
        b = avg_task(baseline_results, task)
        t = avg_task(trained_results, task)
        print(f"  {task.title():<22} {b:>12.4f} {t:>12.4f} {t-b:>+10.4f}")

    wins = sum(1 for b, t in zip(baseline_results, trained_results) if t.final_score > b.final_score)
    print(f"  Wins (trained): {wins}/{num_episodes}")
    print(f"{'='*70}")


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
