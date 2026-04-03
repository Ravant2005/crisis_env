#!/usr/bin/env python3
"""
evaluate_variance.py — Evaluates score variance across multiple episodes.
Runs 10 episodes with the same seed and computes mean/std deviation.
"""

import subprocess
import sys
import statistics

SEED = 42
NUM_EPISODES = 10


def run_episode(seed: int) -> float:
    """Run inference.py and return the final score."""
    env = {"SEED": str(seed), "PYTHONPATH": "."}
    result = subprocess.run(
        ["python3", "inference.py"],
        capture_output=True,
        text=True,
        env={**subprocess.os.environ.copy(), **env},
        cwd=subprocess.os.path.dirname(__file__) or ".",
    )
    
    output = result.stdout
    for line in output.split("\n"):
        if "final_score=" in line:
            for part in line.split("|"):
                if "final" in part and "=" in part:
                    key, val = part.split("=")
                    try:
                        return float(val.strip())
                    except ValueError:
                        pass
    return 0.0


def main():
    print(f"Running variance evaluation with seed={SEED}, {NUM_EPISODES} episodes...\n")
    
    scores = []
    for i in range(NUM_EPISODES):
        print(f"Episode {i+1}/{NUM_EPISODES}...", end=" ")
        score = run_episode(SEED)
        scores.append(score)
        print(f"Score: {score:.4f}")
    
    mean_score = statistics.mean(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    
    print("\n--- Results ---")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Mean Score: {mean_score:.4f}")
    print(f"Std Dev: {std_dev:.4f}")
    print(f"Min: {min(scores):.4f}")
    print(f"Max: {max(scores):.4f}")
    
    if std_dev < 0.01:
        print(f"\nVariance: LOW (excellent reproducibility)")
    elif std_dev < 0.05:
        print(f"\nVariance: ACCEPTABLE")
    else:
        print(f"\nVariance: HIGH (may indicate non-determinism)")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
