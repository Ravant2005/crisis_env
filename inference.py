"""
inference.py — AI Crisis Response & Rescue Coordination Agent.

MANDATORY Submission Format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
import uuid
import requests
from typing import List, Optional, Dict, Any

from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIGURATION  (injected by evaluator)
# ─────────────────────────────────────────────
API_BASE_URL:     str           = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY:          str           = os.getenv("API_KEY", os.getenv("HF_TOKEN", "EMPTY"))
MODEL_NAME:       str           = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN:         Optional[str] = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL:          str           = os.getenv("ENV_URL", "https://praveen4278-crisis-ai-env.hf.space")
SEED:             int           = int(os.getenv("SEED", "42"))
TASK_NAME:        str           = os.getenv("MY_ENV_V4_TASK", "crisis-response")
BENCHMARK:        str           = os.getenv("MY_ENV_V4_BENCHMARK", "openenv")

# ─────────────────────────────────────────────
# LOGGING  (strict format required by evaluator)
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ─────────────────────────────────────────────
# OPENAI CLIENT  (uses evaluator-injected vars)
# ─────────────────────────────────────────────

def make_client() -> OpenAI:
    """Create OpenAI client pointed at the evaluator's LiteLLM proxy."""
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ─────────────────────────────────────────────
# LLM DECISION ENGINE  (called every step)
# ─────────────────────────────────────────────

def _priority_score(threat: Dict[str, Any]) -> float:
    sev = float(threat.get("severity", 1.0))
    pop = int(threat.get("population_at_risk", 1))
    tti = max(int(threat.get("time_to_impact", 1)), 1)
    return (sev * pop) / tti

_ZONE_AFFINITY: Dict[str, List[str]] = {
    "military": ["military_unit", "medical_team"],
    "maritime": ["coast_guard", "rescue_drone"],
    "urban":    ["swat_team", "fire_brigade", "evacuation_bus"],
    "rural":    ["fire_brigade", "medical_team", "rescue_drone"],
}

def _best_resource(threat: Dict[str, Any], resources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    zone = threat.get("zone", "")
    preferred = _ZONE_AFFINITY.get(zone, [])
    available = [r for r in resources if r.get("is_available", False)]
    if not available:
        return None
    return max(available, key=lambda r: float(r.get("effectiveness", 0.5)) + (0.3 if r.get("resource_type") in preferred else 0.0))

def llm_decide(client: OpenAI, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    """
    Ask the LLM what action to take given the current observation.
    This ensures EVERY step makes an LLM call through the proxy.
    Falls back to rule-based if LLM fails.
    """
    threats   = obs.get("threats", [])
    resources = obs.get("resources", [])
    zones     = obs.get("affected_zones", [])
    budget    = int(obs.get("resource_budget_remaining", 0))

    active_threats = [t for t in threats if t.get("status") == "active"]
    active_zones   = [z for z in zones if z.get("is_active", False)]

    # Build a concise state summary for the LLM
    state_summary = (
        f"Step {step}. Budget remaining: {budget}.\n"
        f"Active threats: {[{'id': t['threat_id'], 'type': t['threat_type'], 'sev': t['severity'], 'tti': t['time_to_impact'], 'pop': t['population_at_risk'], 'zone': t['zone']} for t in active_threats]}\n"
        f"Impacted zones needing rescue: {[{'id': z['zone_id'], 'victims': z['total_victims'], 'rescued': z['rescued']} for z in active_zones]}\n"
    )

    prompt = (
        "You are an AI crisis response coordinator. Given the current state, decide the single best action.\n\n"
        f"{state_summary}\n"
        "Available actions: classify, predict, allocate, coordinate, rescue\n\n"
        "Rules:\n"
        "- classify: identify threat type/severity (use threat_id)\n"
        "- predict: forecast TTI and population (use threat_id)\n"
        "- allocate: assign resource to threat (use threat_id + resource_id)\n"
        "- coordinate: rank threats by priority (list threat_ids highest to lowest)\n"
        "- rescue: send units to impacted zone (use zone_id + units 1-5)\n\n"
        "Respond ONLY with a JSON object like one of these:\n"
        '{"action": "classify", "threat_id": 1}\n'
        '{"action": "predict", "threat_id": 1}\n'
        '{"action": "allocate", "threat_id": 1, "resource_id": 2}\n'
        '{"action": "coordinate", "priority_order": [1, 2, 3]}\n'
        '{"action": "rescue", "zone_id": 1, "units": 3}\n'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        decision = json.loads(raw)
        action = decision.get("action", "")

        if action == "classify" and active_threats:
            tid = int(decision.get("threat_id", active_threats[0]["threat_id"]))
            t = next((x for x in active_threats if x["threat_id"] == tid), active_threats[0])
            return {"action_type": "classify", "classification": {"threat_id": t["threat_id"], "predicted_type": t["threat_type"], "predicted_severity": t["severity"]}}

        if action == "predict" and active_threats:
            tid = int(decision.get("threat_id", active_threats[0]["threat_id"]))
            t = next((x for x in active_threats if x["threat_id"] == tid), active_threats[0])
            return {"action_type": "predict", "prediction": {"threat_id": t["threat_id"], "predicted_tti": max(int(t.get("time_to_impact", 5)), 1), "predicted_pop": int(t.get("population_at_risk", 100))}}

        if action == "allocate" and active_threats and resources:
            tid = int(decision.get("threat_id", active_threats[0]["threat_id"]))
            rid = int(decision.get("resource_id", 0))
            t = next((x for x in active_threats if x["threat_id"] == tid), active_threats[0])
            r = next((x for x in resources if x.get("is_available") and x["resource_id"] == rid), None) or _best_resource(t, resources)
            if r:
                return {"action_type": "allocate", "allocation": {"threat_id": t["threat_id"], "resource_id": r["resource_id"]}}

        if action == "coordinate" and threats:
            order = decision.get("priority_order", [t["threat_id"] for t in sorted(active_threats, key=_priority_score, reverse=True)])
            return {"action_type": "coordinate", "coordination": {"priority_order": order}}

        if action == "rescue" and active_zones and budget > 0:
            zid   = int(decision.get("zone_id", active_zones[0]["zone_id"]))
            units = min(int(decision.get("units", 3)), budget, 5)
            z = next((x for x in active_zones if x["zone_id"] == zid), active_zones[0])
            return {"action_type": "rescue", "rescue": {"zone_id": z["zone_id"], "rescue_units_to_send": units}}

    except Exception as e:
        print(f"[LLM_FALLBACK] {e}", flush=True)

    # ── Rule-based fallback ──────────────────────────────────────────────────
    return _rule_based_action(active_threats, active_zones, resources, budget)


def _rule_based_action(active_threats, active_zones, resources, budget) -> Dict[str, Any]:
    """Fallback rule-based action when LLM fails."""
    ranked = sorted(active_threats, key=_priority_score, reverse=True)

    for t in ranked:
        if t.get("predicted_severity") is None:
            return {"action_type": "classify", "classification": {"threat_id": t["threat_id"], "predicted_type": t["threat_type"], "predicted_severity": t["severity"]}}

    for t in ranked:
        if t.get("predicted_tti") is None:
            return {"action_type": "predict", "prediction": {"threat_id": t["threat_id"], "predicted_tti": max(int(t.get("time_to_impact", 5)), 1), "predicted_pop": int(t.get("population_at_risk", 100))}}

    if len(ranked) >= 2:
        return {"action_type": "coordinate", "coordination": {"priority_order": [t["threat_id"] for t in ranked]}}

    for t in ranked:
        r = _best_resource(t, resources)
        if r and budget > 0:
            return {"action_type": "allocate", "allocation": {"threat_id": t["threat_id"], "resource_id": r["resource_id"]}}

    if active_zones and budget > 0:
        z = max(active_zones, key=lambda x: x.get("total_victims", 0) - x.get("rescued", 0))
        return {"action_type": "rescue", "rescue": {"zone_id": z["zone_id"], "rescue_units_to_send": min(5, budget)}}

    return {"action_type": "skip"}

# ─────────────────────────────────────────────
# HTTP CLIENT  (environment server)
# ─────────────────────────────────────────────

def _env_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h

async def http_reset(seed: int, session_id: str) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/reset",
        json={"seed": seed, "difficulty": "medium", "session_id": session_id},
        headers=_env_headers(), timeout=30,
    ).json())

async def http_step(action: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/step",
        json={"action": action, "session_id": session_id},
        headers=_env_headers(), timeout=30,
    ).json())

async def http_state(session_id: str) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.get(
        f"{ENV_URL}/state",
        params={"session_id": session_id},
        headers=_env_headers(), timeout=30,
    ).json())

# ─────────────────────────────────────────────
# MAIN EPISODE LOOP
# ─────────────────────────────────────────────

async def main() -> None:
    session_id = f"episode_{uuid.uuid4().hex[:8]}"
    client     = make_client()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = await http_reset(seed=SEED, session_id=session_id)
        obs        = reset_resp.get("observation", {})
        rewards    = []
        step       = 0
        done       = False

        while not done:
            step += 1

            # LLM called on EVERY step — always hits the proxy
            action_payload = llm_decide(client, obs, step)

            res    = await http_step(action_payload, session_id)
            reward = res.get("reward", 0.0)
            done   = res.get("done", False)
            obs    = res.get("observation", obs)
            rewards.append(reward)

            log_step(step, action_payload.get("action_type", "unknown"), reward, done, None)

            if step > 50:  # safety guard
                break

        state       = await http_state(session_id)
        final_score = state.get("final_score", 0.0)
        log_end(done, step, final_score, rewards)

    except Exception as e:
        log_end(False, 0, 0.0, [])
        print(f"[ERROR] {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
