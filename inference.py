"""
Inference Script — CrisisAI: AI Crisis Response & Rescue Coordination
===================================
MANDATORY environment variables:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  The name of the local Docker image for the environment.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
import uuid
import textwrap
import requests
from typing import List, Optional, Dict, Any

from openai import OpenAI

# ─────────────────────────────────────────────
# MANDATORY CONFIGURATION
# ─────────────────────────────────────────────
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME       = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME        = os.getenv("MY_ENV_V4_TASK",      "crisis-response")
BENCHMARK        = os.getenv("MY_ENV_V4_BENCHMARK", "openenv")
SEED             = int(os.getenv("SEED", "42"))
ENV_URL          = os.getenv("ENV_URL", "https://praveen4278-crisis-ai-env.hf.space")

MAX_STEPS             = 30
SUCCESS_SCORE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# LOGGING  (exact format required)
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ─────────────────────────────────────────────
# ENVIRONMENT HTTP CLIENT
# ─────────────────────────────────────────────

def _env_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h

async def env_reset(session_id: str) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/reset",
        json={"seed": SEED, "difficulty": "medium", "session_id": session_id},
        headers=_env_headers(), timeout=30,
    ).json())

async def env_step(action: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/step",
        json={"action": action, "session_id": session_id},
        headers=_env_headers(), timeout=30,
    ).json())

async def env_state(session_id: str) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.get(
        f"{ENV_URL}/state",
        params={"session_id": session_id},
        headers=_env_headers(), timeout=30,
    ).json())

# ─────────────────────────────────────────────
# LLM DECISION  (called every step via proxy)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI crisis response coordinator managing emergency threats.
    Each turn you must decide one action to take based on the current state.
    Your goal is to classify threats, predict impacts, allocate resources,
    coordinate responses, and rescue victims to maximize lives saved.
    Reply with exactly one JSON action object — no explanation, just JSON.
""").strip()

def get_llm_action(client: OpenAI, obs: Dict[str, Any], step: int, history: List[str]) -> Dict[str, Any]:
    threats   = obs.get("threats", [])
    resources = obs.get("resources", [])
    zones     = obs.get("affected_zones", [])
    budget    = int(obs.get("resource_budget_remaining", 0))

    active_threats = [t for t in threats if t.get("status") == "active"]
    active_zones   = [z for z in zones if z.get("is_active", False)]
    avail_res      = [r for r in resources if r.get("is_available", False)]

    user_prompt = textwrap.dedent(f"""
        Step {step}. Budget: {budget}.
        Active threats: {[{'id': t['threat_id'], 'type': t['threat_type'], 'sev': t['severity'], 'tti': t['time_to_impact'], 'pop': t['population_at_risk'], 'zone': t['zone']} for t in active_threats]}
        Available resources: {[{'id': r['resource_id'], 'type': r['resource_type'], 'eff': r['effectiveness']} for r in avail_res]}
        Zones needing rescue: {[{'id': z['zone_id'], 'victims': z['total_victims'], 'rescued': z['rescued']} for z in active_zones]}
        Recent actions: {history[-3:]}

        Choose ONE action. Reply with exactly one of these JSON formats:
        {{"action_type": "classify", "classification": {{"threat_id": <id>, "predicted_type": "<type>", "predicted_severity": <0-10>}}}}
        {{"action_type": "predict", "prediction": {{"threat_id": <id>, "predicted_tti": <int>, "predicted_pop": <int>}}}}
        {{"action_type": "allocate", "allocation": {{"threat_id": <id>, "resource_id": <id>}}}}
        {{"action_type": "coordinate", "coordination": {{"priority_order": [<id1>, <id2>, ...]}}}}
        {{"action_type": "rescue", "rescue": {{"zone_id": <id>, "rescue_units_to_send": <1-5>}}}}
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
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Extract JSON from response
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            action = json.loads(text[start:end])
            if "action_type" in action:
                return action
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)

    # Rule-based fallback
    return _fallback_action(active_threats, active_zones, avail_res, budget)

def _fallback_action(active_threats, active_zones, avail_res, budget) -> Dict[str, Any]:
    def priority(t):
        return (float(t.get("severity", 1)) * int(t.get("population_at_risk", 1))) / max(int(t.get("time_to_impact", 1)), 1)

    ranked = sorted(active_threats, key=priority, reverse=True)

    for t in ranked:
        if t.get("predicted_severity") is None:
            return {"action_type": "classify", "classification": {"threat_id": t["threat_id"], "predicted_type": t["threat_type"], "predicted_severity": t["severity"]}}
    for t in ranked:
        if t.get("predicted_tti") is None:
            return {"action_type": "predict", "prediction": {"threat_id": t["threat_id"], "predicted_tti": max(int(t.get("time_to_impact", 5)), 1), "predicted_pop": int(t.get("population_at_risk", 100))}}
    if len(ranked) >= 2:
        return {"action_type": "coordinate", "coordination": {"priority_order": [t["threat_id"] for t in ranked]}}
    for t in ranked:
        if avail_res and budget > 0:
            return {"action_type": "allocate", "allocation": {"threat_id": t["threat_id"], "resource_id": avail_res[0]["resource_id"]}}
    if active_zones and budget > 0:
        z = max(active_zones, key=lambda x: x.get("total_victims", 0) - x.get("rescued", 0))
        return {"action_type": "rescue", "rescue": {"zone_id": z["zone_id"], "rescue_units_to_send": min(5, budget)}}
    return {"action_type": "skip"}

# ─────────────────────────────────────────────
# MAIN EPISODE LOOP
# ─────────────────────────────────────────────

async def main() -> None:
    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    session_id = f"episode_{uuid.uuid4().hex[:8]}"

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result      = await env_reset(session_id)
        obs         = result.get("observation", {})
        done        = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_llm_action(client, obs, step, history)

            result  = await env_step(action, session_id)
            reward  = float(result.get("reward", 0.0))
            done    = bool(result.get("done", False))
            obs     = result.get("observation", obs)
            error   = None

            rewards.append(reward)
            steps_taken = step
            action_str  = action.get("action_type", "unknown")
            history.append(f"step={step} action={action_str} reward={reward:.2f}")

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        state = await env_state(session_id)
        score = float(state.get("final_score", 0.0))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
