#!/usr/bin/env python3
"""
test_determinism.py — Tests that the environment produces deterministic results.
Runs the same episode 3 times with the same seed and verifies identical outputs.
"""

import os
import subprocess
import sys
import json

SEED = 42
NUM_RUNS = 3


def run_episode(seed: int) -> dict:
    """Run inference.py and capture scores."""
    env = {"SEED": str(seed), "PYTHONPATH": "."}
    result = subprocess.run(
        ["python3", "inference.py"],
        capture_output=True,
        text=True,
        env={**os.environ.copy(), **env},
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    
    output = result.stdout
    for line in output.split("\n"):
        if "[SCORE]" in line:
            scores = {}
            for part in line.split("|"):
                if "=" in part:
                    key, val = part.split("=")
                    key = key.strip().replace("[SCORE] ", "")
                    try:
                        scores[key] = float(val.strip())
                    except ValueError:
                        pass
            return scores
    return {}


def main():
    print(f"Running determinism test with seed={SEED}, {NUM_RUNS} times...\n")
    
    all_scores = []
    for i in range(NUM_RUNS):
        print(f"Run {i+1}/{NUM_RUNS}...")
        scores = run_episode(SEED)
        all_scores.append(scores)
        print(f"  Final Score: {scores.get('final', 'N/A')}")
    
    if len(all_scores) < NUM_RUNS:
        print("\n[ERROR] Not all runs completed successfully")
        sys.exit(1)
    
    print("\n--- Results ---")
    for i, scores in enumerate(all_scores):
        print(f"Run {i+1}: {scores}")
    
    baseline = all_scores[0]
    all_match = True
    
    for key in baseline:
        values = [s.get(key, None) for s in all_scores]
        if None in values:
            all_match = False
            continue
        if not all(abs(v - values[0]) < 0.0001 for v in values):
            all_match = False
            print(f"MISMATCH: {key} -> {values}")
    
    if all_match:
        print("\nDeterministic: PASS")
        sys.exit(0)
    else:
        print("\nDeterministic: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
