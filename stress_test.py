#!/usr/bin/env python3
"""
stress_test.py — Runs inference.py multiple times and computes performance statistics.
"""

import subprocess
import re
import statistics
import sys
import os
import time

NUM_RUNS = 20
API_URL = "http://localhost:8000"


def run_inference(seed: int) -> float | None:
    """Run inference.py with a specific seed and extract final score."""
    env = {
        "ENV_API_URL": API_URL,
        "SEED": str(seed),
        "USE_LLM": "false",
    }
    
    result = subprocess.run(
        ["python3", "inference.py"],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    
    if result.returncode != 0:
        print(f"[WARN] Run with seed={seed} failed: {result.stderr[:200]}")
        return None
    
    output = result.stdout
    
    score_match = re.search(r"\[SCORE\].*final=([0-9.]+)", output)
    if score_match:
        return float(score_match.group(1))
    
    print(f"[WARN] Could not extract score from run seed={seed}")
    return None


def run_stress_test(num_runs: int = NUM_RUNS) -> dict:
    """Run multiple inference episodes and compute statistics."""
    scores = []
    failed_runs = 0
    
    print(f"Starting stress test with {num_runs} runs...")
    print("-" * 40)
    
    for i in range(num_runs):
        seed = 42 + i
        print(f"Run {i+1}/{num_runs} (seed={seed})...", end=" ", flush=True)
        
        score = run_inference(seed)
        if score is not None:
            scores.append(score)
            print(f"score={score:.4f}")
        else:
            failed_runs += 1
            print("FAILED")
        
        time.sleep(0.1)
    
    if not scores:
        print("\n[ERROR] No successful runs completed!")
        return {"error": "No successful runs"}
    
    mean_score = statistics.mean(scores)
    min_score = min(scores)
    max_score = max(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    
    result = {
        "num_runs": num_runs,
        "successful_runs": len(scores),
        "failed_runs": failed_runs,
        "mean_score": mean_score,
        "min_score": min_score,
        "max_score": max_score,
        "std_dev": std_dev,
    }
    
    return result


def print_results(result: dict):
    """Print formatted test results."""
    print("\n" + "=" * 40)
    print("STRESS TEST RESULTS")
    print("=" * 40)
    print(f"Runs: {result['num_runs']}")
    print(f"Successful: {result['successful_runs']}")
    print(f"Failed: {result['failed_runs']}")
    print("-" * 40)
    print(f"Mean Score: {result['mean_score']:.2f}")
    print(f"Min Score: {result['min_score']:.2f}")
    print(f"Max Score: {result['max_score']:.2f}")
    print(f"Std Dev: {result['std_dev']:.2f}")
    print("=" * 40)
    
    mean_ok = result['mean_score'] > 0.80
    std_ok = result['std_dev'] < 0.10
    
    if mean_ok and std_ok:
        print("✅ PASS - Mean > 0.80 and Std Dev < 0.10")
        return 0
    else:
        print("❌ FAIL")
        if not mean_ok:
            print(f"   - Mean score {result['mean_score']:.2f} <= 0.80")
        if not std_ok:
            print(f"   - Std dev {result['std_dev']:.2f} >= 0.10")
        return 1


def main():
    result = run_stress_test(NUM_RUNS)
    return print_results(result)


if __name__ == "__main__":
    sys.exit(main())