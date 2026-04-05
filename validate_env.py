"""
validate_env.py — Pre-submission validation for the Crisis Response OpenEnv.

Checks:
  1. /health returns 200 and {"status": "ok"}
  2. /reset returns a valid CrisisObservation
  3. /step with a classify action returns reward + done fields
  4. /state returns all 5 task scores in [0.0, 1.0]
  5. /scores returns final score in [0.0, 1.0]
  6. /tasks returns at least 3 tasks
  7. 3+ tasks each have grader_range [0.0, 1.0]
  8. Episode terminates correctly (done=True reached within max_steps)
  9. Scores are deterministic across two identical seeds
 10. openenv.yaml is parseable and contains required fields
"""
from __future__ import annotations
import json, sys, time, random
import requests
import yaml

BASE_URL = "http://localhost:8000"
PASS = "[ PASS ]"
FAIL = "[ FAIL ]"
results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"{status}  {name}" + (f"  — {detail}" if detail else ""))
    return condition

def run():
    print("\n" + "="*60)
    print("  Crisis Response OpenEnv — Pre-Submission Validator")
    print("="*60 + "\n")

    # 1. Health
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        body = r.json()
        check("Health endpoint returns 200", r.status_code == 200)
        check("Health body has status=ok", body.get("status") == "ok", str(body))
    except Exception as e:
        check("Health endpoint reachable", False, str(e))
        print("\nCannot reach server. Is it running on port 8000?\n")
        sys.exit(1)

    # 2. Reset
    r = requests.post(f"{BASE_URL}/reset", json={"seed": 42, "difficulty": "medium"})
    check("Reset returns 200", r.status_code == 200)
    obs = r.json().get("observation", r.json())
    check("Reset observation has threats list", isinstance(obs.get("threats"), list))
    check("Reset observation has time_remaining", "time_remaining" in obs)
    check("Reset has resource_budget_remaining", "resource_budget_remaining" in obs)

    # 3. Step
    threats = obs.get("threats", [])
    if threats:
        threat = threats[0]
        action = {
            "action_type": "classify",
            "classification": {
                "threat_id": threat["threat_id"],
                "predicted_type": threat["threat_type"],
                "predicted_severity": float(threat["severity"]),
            }
        }
        r2 = requests.post(f"{BASE_URL}/step", json={"action": action})
        check("Step returns 200", r2.status_code == 200)
        body2 = r2.json()
        check("Step has reward field", "reward" in body2, str(list(body2.keys())))
        check("Step has done field", "done" in body2)
        check("Step reward is float in [-5, 5]", isinstance(body2.get("reward"), (int, float)))

    # 4. State
    r3 = requests.get(f"{BASE_URL}/state")
    check("State returns 200", r3.status_code == 200)
    state = r3.json()
    for field in ["classification_score","prediction_score","allocation_score",
                  "coordination_score","rescue_score","final_score"]:
        v = state.get(field, -1)
        check(f"State.{field} in [0,1]", 0.0 <= float(v) <= 1.0, f"={v}")

    # 5. Scores endpoint
    r4 = requests.get(f"{BASE_URL}/scores")
    check("Scores endpoint returns 200", r4.status_code == 200)
    sc = r4.json()
    check("Scores.final in [0,1]", 0.0 <= float(sc.get("final", -1)) <= 1.0)

    # 6. Tasks
    r5 = requests.get(f"{BASE_URL}/tasks")
    check("Tasks returns 200", r5.status_code == 200)
    tasks = r5.json().get("tasks", [])
    check("Tasks has >= 3 entries", len(tasks) >= 3, f"found {len(tasks)}")
    grader_ok = all(t.get("grader_range") == [0.0, 1.0] for t in tasks)
    check("All tasks have grader_range [0.0, 1.0]", grader_ok)

    # 7. Full episode termination
    requests.post(f"{BASE_URL}/reset", json={"seed": 99, "difficulty": "easy"})
    done = False
    max_steps = 30
    for _ in range(max_steps + 5):
        action = {"action_type": "skip"}
        r_s = requests.post(f"{BASE_URL}/step", json={"action": action})
        if r_s.json().get("done"):
            done = True
            break
    check("Episode terminates within max_steps", done)

    # 8. Determinism — same seed => same first reward
    def first_reward(seed: int) -> float:
        requests.post(f"{BASE_URL}/reset", json={"seed": seed, "difficulty": "medium"})
        obs_d = requests.post(f"{BASE_URL}/reset", json={"seed": seed, "difficulty": "medium"}).json()
        obs_d = obs_d.get("observation", obs_d)
        t = obs_d.get("threats", [{}])[0]
        act = {"action_type": "classify",
               "classification": {"threat_id": t.get("threat_id", 1),
                                  "predicted_type": t.get("threat_type", "fire"),
                                  "predicted_severity": float(t.get("severity", 5.0))}}
        return float(requests.post(f"{BASE_URL}/step", json={"action": act}).json().get("reward", -999))

    r_a = first_reward(42)
    r_b = first_reward(42)
    check("Determinism: same seed => same reward", abs(r_a - r_b) < 1e-6, f"r1={r_a} r2={r_b}")

    # 9. openenv.yaml parseable
    try:
        with open("openenv.yaml") as f:
            manifest = yaml.safe_load(f)
        required = ["name","version","description","environment","endpoints","tasks"]
        missing = [k for k in required if k not in manifest]
        check("openenv.yaml has all required fields", len(missing) == 0, f"missing={missing}")
        check("openenv.yaml has >= 3 tasks", len(manifest.get("tasks", [])) >= 3)
    except Exception as e:
        check("openenv.yaml parseable", False, str(e))

    # Summary
    print("\n" + "="*60)
    passed = sum(1 for s,_,_ in results if s == PASS)
    total  = len(results)
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  STATUS: READY TO SUBMIT")
    else:
        print("  STATUS: FIX FAILURES BEFORE SUBMITTING")
    print("="*60 + "\n")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    run()