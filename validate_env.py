#!/usr/bin/env python3
"""
validate_env.py — Validates the OpenEnv environment configuration.
Runs: openenv validate
"""

import subprocess
import sys


def main():
    result = subprocess.run(["openenv", "validate"], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Validation failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print("\n[SUCCESS] OpenEnv validation passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
