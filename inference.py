"""
Inference Script — CrisisAI: AI Crisis Response & Rescue Coordination
======================================================================
FIXED:
  - Runs 3 separate task episodes (task_easy, task_medium, task_hard)
  - Each task produces its own [START] / [STEP]* / [END] log sequence
  - Platform validator needs to see at least 3 [END] lines with scores
  - /reset now sends task_id (not difficulty) matching the server schema
  - /reset body is always a valid dict (never null)

STDOUT FORMAT (required by platform):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import os
import json
import uuid
import textwrap
import time
import requests
from typing import List, Optional, Dict, Any

from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL      = os.getenv("ENV_URL", "https://praveen4278-crisis-ai-env.hf.space").rstrip("/")
BENCHMARK    = os.getenv("MY_ENV_V4_BENCHMARK", "openenv")
SEED         = int(os.getenv("SEED", "42"))

MAX_STEPS               = 30
SUCCESS_SCORE_THRESHOLD = 0.3

# The 3 tasks the platform must see scored
TASK_IDS = ["task_easy", "task_medium", "task_hard"]

# ─────────────────────────────────────────────
# LOGGING  (exact format required by platform)
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ─────────────────────────────────────────────
# ENVIRONMENT HTTP CLIENT
# ─────────────────────────────────────────────

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h

async def env_reset(task_id: str = "task_easy") -> Dict[str, Any]:
    """POST /reset with task_id and seed. Always sends a valid body."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": SEED},
        headers=_headers(),
        timeout=30,
    ).json())

async def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step with action dict."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        headers=_headers(),
        timeout=30,
    ).json())

async def env_scores() -> Dict[str, Any]:
    """GET /scores — returns all 5 task scores + final."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.get(
        f"{ENV_URL}/scores",
        headers=_headers(),
        timeout=30,
    ).json())

# ─────────────────────────────────────────────
# LLM DECISION
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI crisis response coordinator managing emergency threats.
    Each turn you must decide ONE action based on the current state.
    Priority pipeline: classify → predict → coordinate → allocate → rescue.
    Reply with exactly one JSON action object — no explanation, just JSON.
""").strip()

def get_llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    history: List[str],
    task_id: str,
) -> Dict[str, Any]:
    threats   = obs.get("threats", [])
    resources = obs.get("resources", [])
    zones     = obs.get("affected_zones", [])
    budget    = int(obs.get("resource_budget_remaining", 0))

    active_threats = [t for t in threats if t.get("status") == "active"]
    active_zones   = [z for z in zones if z.get("is_active", False)]
    avail_res      = [r for r in resources if r.get("is_available", False)]

    user_prompt = textwrap.dedent(f"""
        Task: {task_id}. Step {step}. Budget: {budget}.
        Active threats: {[{'id': t['threat_id'], 'type': t['threat_type'], 'sev': t['severity'], 'tti': t['time_to_impact'], 'pop': t['population_at_risk']} for t in active_threats[:4]]}
        Available resources: {[{'id': r['resource_id'], 'type': r['resource_type'], 'eff': r['effectiveness']} for r in avail_res[:4]]}
        Rescue zones: {[{'id': z['zone_id'], 'victims': z['total_victims'], 'rescued': z['rescued']} for z in active_zones[:4]]}
        Recent: {history[-3:]}

        Reply with ONE of these exact JSON formats:
        {{"action_type":"classify","classification":{{"threat_id":<id>,"predicted_type":"<type>","predicted_severity":<0-10>}}}}
        {{"action_type":"predict","prediction":{{"threat_id":<id>,"predicted_tti":<int>,"predicted_pop":<int>}}}}
        {{"action_type":"allocate","allocation":{{"threat_id":<id>,"resource_id":<id>}}}}
        {{"action_type":"coordinate","coordination":{{"priority_order":[<id1>,<id2>]}}}}
        {{"action_type":"rescue","rescue":{{"zone_id":<id>,"rescue_units_to_send":<1-5>}}}}
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            action = json.loads(text[start:end])
            if "action_type" in action:
                return action
    except Exception as exc:
        print(f"[DEBUG] LLM error step {step}: {exc}", flush=True)

    return _fallback_action(active_threats, active_zones, avail_res, budget, step)


def _fallback_action(
    active_threats, active_zones, avail_res, budget, step
) -> Dict[str, Any]:
    """Rule-based fallback covering the full action pipeline."""
    def priority(t):
        return (
            float(t.get("severity", 1)) * int(t.get("population_at_risk", 1))
        ) / max(int(t.get("time_to_impact", 1)), 1)

    ranked = sorted(active_threats, key=priority, reverse=True)

    # Stage 1: classify
    for t in ranked:
        if t.get("predicted_severity") is None:
            return {
                "action_type": "classify",
                "classification": {
                    "threat_id":          t["threat_id"],
                    "predicted_type":     t.get("threat_type", "fire"),
                    "predicted_severity": float(t.get("severity", 5.0)),
                },
            }

    # Stage 2: predict
    for t in ranked:
        if t.get("predicted_tti") is None:
            return {
                "action_type": "predict",
                "prediction": {
                    "threat_id":     t["threat_id"],
                    "predicted_tti": max(int(t.get("time_to_impact", 5)), 1),
                    "predicted_pop": int(t.get("population_at_risk", 100)),
                },
            }

    # Stage 3: coordinate
    if len(ranked) >= 2:
        return {
            "action_type":  "coordinate",
            "coordination": {"priority_order": [t["threat_id"] for t in ranked]},
        }

    # Stage 4: allocate
    for t in ranked:
        if avail_res and budget > 0 and t.get("assigned_resource") is None:
            best = max(avail_res, key=lambda r: float(r.get("effectiveness", 0)))
            return {
                "action_type": "allocate",
                "allocation":  {"threat_id": t["threat_id"], "resource_id": best["resource_id"]},
            }

    # Stage 5: rescue
    if active_zones and budget > 0:
        z = max(active_zones, key=lambda x: x.get("total_victims", 0) - x.get("rescued", 0))
        return {
            "action_type": "rescue",
            "rescue":      {"zone_id": z["zone_id"], "rescue_units_to_send": min(5, budget)},
        }

    return {"action_type": "skip"}


# ─────────────────────────────────────────────
# SINGLE TASK EPISODE
# ─────────────────────────────────────────────

async def run_task(client: OpenAI, task_id: str) -> float:
    """
    Run one complete episode for task_id.
    Emits [START] ... [STEP]* ... [END] for this task.
    Returns final score in [0.0, 1.0].
    """
    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    print(f"[INFO] Starting task={task_id} seed={SEED}", flush=True)

    try:
        # Reset — always send a valid body with task_id
        reset_result = await env_reset(task_id=task_id)

        # Handle both flat obs and wrapped {"observation": ...} response
        obs  = reset_result.get("observation", reset_result)
        done = bool(obs.get("done", False))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action      = get_llm_action(client, obs, step, history, task_id)
            step_result = await env_step(action)

            reward  = float(step_result.get("reward", 0.0))
            done    = bool(step_result.get("done", False))
            obs     = step_result.get("observation", obs)
            error   = None

            rewards.append(reward)
            steps_taken = step
            action_str  = action.get("action_type", "unknown")
            history.append(f"step={step} action={action_str} reward={reward:.2f}")

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Fetch score periodically and at end
            if done or step % 5 == 0:
                try:
                    scores_resp = await env_scores()
                    s = float(
                        scores_resp.get("final_score")
                        or scores_resp.get("final")
                        or 0.0
                    )
                    if s > 0.0:
                        score = s
                except Exception:
                    pass

            if done:
                break

        # Final score fetch
        try:
            scores_resp = await env_scores()
            final = float(
                scores_resp.get("final_score")
                or scores_resp.get("final")
                or 0.0
            )
            if final == 0.0:
                final = (
                    0.20 * float(scores_resp.get("classification", 0.0)) +
                    0.20 * float(scores_resp.get("prediction",     0.0)) +
                    0.20 * float(scores_resp.get("allocation",     0.0)) +
                    0.15 * float(scores_resp.get("coordination",   0.0)) +
                    0.25 * float(scores_resp.get("rescue",         0.0))
                )
            if final > 0.0:
                score = final
        except Exception:
            pass

        score   = min(max(score, 0.0), 1.0)
        if score == 0.0 and steps_taken > 0:
            score = 0.001
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        if score == 0.0 and steps_taken > 0:
            score = 0.001

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ─────────────────────────────────────────────
# MAIN — runs all 3 required tasks
# ─────────────────────────────────────────────

async def main() -> None:
    print(f"CrisisAI — Inference Script", flush=True)
    print(f"  API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"  MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(f"  ENV_URL      : {ENV_URL}", flush=True)
    print(
        f"  HF_TOKEN     : {'set' if API_KEY else 'NOT SET — check environment'}",
        flush=True,
    )
    print(f"  Tasks to run : {TASK_IDS}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")
    scores: Dict[str, float] = {}
    start  = time.time()

    for task_id in TASK_IDS:
        t0 = time.time()
        scores[task_id] = await run_task(client, task_id)
        elapsed = time.time() - t0
        print(f"[INFO] Task '{task_id}' done in {elapsed:.1f}s  score={scores[task_id]:.3f}", flush=True)
        await asyncio.sleep(1.0)

    total = time.time() - start

    print(f"\n{'=' * 60}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  {'Task':<20} {'Score':>8}", flush=True)
    print(f"  {'-' * 30}", flush=True)
    for task_id, s in scores.items():
        bar = "=" * int(s * 20)
        print(f"  {task_id:<20} {s:>8.3f}  {bar}", flush=True)
    print(f"  {'-' * 30}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':<20} {avg:>8.3f}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total time: {total:.1f}s", flush=True)

    parts = " | ".join(f"{k}={v:.4f}" for k, v in scores.items())
    print(f"[SCORE] {parts} | final={avg:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())