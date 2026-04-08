"""
inference.py — AI Crisis Response & Rescue Coordination Agent.
Deterministic baseline agent using a prioritized pipeline.

Strategy:
    Pipeline per threat: CLASSIFY -> PREDICT -> ALLOCATE
    Secondary tasks: COORDINATE -> RESCUE

MANDATORY Submission Format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
import time
import uuid
import textwrap
import requests
from typing import List, Optional, Dict, Any, Tuple

from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# ── Mandatory submission variables ──────────────────────────────────────────
# API_BASE_URL : LLM API endpoint (OpenAI-compatible)
# MODEL_NAME   : model identifier for LLM calls
# HF_TOKEN     : Hugging Face / API key (no default)
# LOCAL_IMAGE_NAME : optional, for from_docker_image()
API_BASE_URL:     str           = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:       str           = os.getenv("MODEL_NAME",  "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN:         Optional[str] = os.getenv("HF_TOKEN")           # no default
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")   # for from_docker_image()

# ── Environment server URL (separate from LLM API) ───────────────────────────
# The evaluator sets this to point at the running OpenEnv server.
ENV_URL: str = os.getenv("ENV_URL", "https://praveen4278-crisis-ai-env.hf.space")

# ── Project-specific variables ────────────────────────────────────────────────
TASK_NAME:       str  = os.getenv("MY_ENV_V4_TASK",      "crisis-response")
BENCHMARK:       str  = os.getenv("MY_ENV_V4_BENCHMARK", "openenv")
SEED:            int  = int(os.getenv("SEED", "42"))
USE_LLM:         bool = os.getenv("USE_LLM", "false").lower() == "true"
RECOORD_INTERVAL      = 12

# ─────────────────────────────────────────────
# LOGGING UTILS
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─────────────────────────────────────────────
# HTTP CLIENT (Environment)
# ─────────────────────────────────────────────

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h

async def http_reset(seed: int = SEED, difficulty: str = "medium", session_id: str = "test_session") -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/reset",
        json={"seed": seed, "difficulty": difficulty, "session_id": session_id},
        headers=_headers(),
        timeout=30,
    ).json())

async def http_step(action: Dict[str, Any], session_id: str = "test_session") -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.post(
        f"{ENV_URL}/step",
        json={"action": action, "session_id": session_id},
        headers=_headers(),
        timeout=30,
    ).json())

async def http_state(session_id: str = "test_session") -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: requests.get(
        f"{ENV_URL}/state",
        params={"session_id": session_id},
        headers=_headers(),
        timeout=30,
    ).json())

# ─────────────────────────────────────────────
# RULE-BASED DECISION ENGINE
# ─────────────────────────────────────────────

_ZONE_AFFINITY: Dict[str, List[str]] = {
    "military": ["military_unit", "medical_team"],
    "maritime": ["coast_guard", "rescue_drone"],
    "urban":    ["swat_team", "fire_brigade", "evacuation_bus"],
    "rural":    ["fire_brigade", "medical_team", "rescue_drone"],
}

def _priority_score(threat: Dict[str, Any]) -> float:
    sev = float(threat.get("severity", 1.0))
    pop = int(threat.get("population_at_risk", 1))
    tti = max(int(threat.get("time_to_impact", 1)), 1)
    return (sev * pop) / tti

def _best_resource(threat: Dict[str, Any], resources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    zone = threat.get("zone", "")
    preferred = _ZONE_AFFINITY.get(zone, [])
    available = [r for r in resources if r.get("is_available", False)]
    if not available: return None
    def resource_score(r: Dict[str, Any]) -> float:
        base = float(r.get("effectiveness", 0.5))
        bonus = 0.3 if r.get("resource_type") in preferred else 0.0
        return base + bonus
    return max(available, key=resource_score)

def _classify_action(threat: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_type": "classify",
        "classification": {
            "threat_id": threat["threat_id"],
            "predicted_type": threat["threat_type"],
            "predicted_severity": threat["severity"],
        },
    }

def _predict_action(threat: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_type": "predict",
        "prediction": {
            "threat_id": threat["threat_id"],
            "predicted_tti": max(int(threat.get("time_to_impact", 5)), 1),
            "predicted_pop": int(threat.get("population_at_risk", 100)),
        },
    }

def _allocate_action(threat: Dict[str, Any], resources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    res = _best_resource(threat, resources)
    if res is None: return None
    return {
        "action_type": "allocate",
        "allocation": {"threat_id": threat["threat_id"], "resource_id": res["resource_id"]},
    }

async def _llm_suggest_priority(threats: List[Dict[str, Any]], client: OpenAI) -> List[int]:
    active = [t for t in threats if t.get("status") == "active"]
    if not active: return [t["threat_id"] for t in threats]
    prompt = (
        "You are a crisis coordinator. Rank the following threats from highest to lowest priority.\n"
        "Priority formula: severity × population_at_risk / time_to_impact\n\n"
        "Threats:\n"
        + "\n".join(f"- threat_id={t['threat_id']}, type={t['threat_type']}, severity={t['severity']}, population={t['population_at_risk']}, tti={t['time_to_impact']}" for t in active)
        + "\n\nRespond ONLY with a JSON array of threat_ids in priority order."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,
        )
        ids = json.loads(response.choices[0].message.content.strip())
        if isinstance(ids, list): return ids
    except:
        pass
    return [t["threat_id"] for t in sorted(active, key=_priority_score, reverse=True)]

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

async def main() -> None:
    session_id = f"episode_{uuid.uuid4().hex[:8]}"
    
    # OpenAI client — uses API_BASE_URL (LLM endpoint) + HF_TOKEN per submission spec
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN if HF_TOKEN else "EMPTY")

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = await http_reset(seed=SEED, session_id=session_id)
        obs = reset_resp.get("observation", {})
        threats = obs.get("threats", [])
        resources = obs.get("resources", [])
        zones = obs.get("affected_zones", [])
        budget = int(obs.get("resource_budget_remaining", 99))
        
        classified, predicted, allocated = set(), set(), set()
        coordinated = False
        last_coord_step = -1
        rewards = []
        step = 0
        done = False

        while not done:
            step += 1
            active_threats = [t for t in threats if t.get("status") == "active"]
            active_zones = [z for z in zones if z.get("is_active", False)]
            
            # PHASE ENGINE
            phase = None
            if active_threats:
                active_threats = sorted(active_threats, key=lambda t: t.get("time_to_impact", 99))
                
                # Coordination check
                ready_threats = [t for t in threats if t["threat_id"] in classified and t["threat_id"] in predicted]
                if not coordinated and len(ready_threats) >= 2:
                    phase = "coordinate"
                
                if phase is None:
                    for t in active_threats:
                        tid = t["threat_id"]
                        if tid not in classified:
                            phase, target_threat = "classify", t
                            break
                        elif tid not in predicted:
                            phase, target_threat = "predict", t
                            break
                        elif tid not in allocated and budget > 0:
                            phase, target_threat = "allocate", t
                            break
                
                if phase is None:
                    if (step - last_coord_step >= RECOORD_INTERVAL) and len(active_threats) >= 2:
                        phase = "coordinate"
                    elif active_zones and budget > 0:
                        phase = "rescue"
            elif active_zones and budget > 0:
                phase = "rescue"
            
            if phase is None: break

            # BUILD ACTION
            action_payload = None
            if phase == "classify":
                action_payload = _classify_action(target_threat)
            elif phase == "predict":
                action_payload = _predict_action(target_threat)
            elif phase == "allocate":
                action_payload = _allocate_action(target_threat, resources)
            elif phase == "coordinate":
                order = await _llm_suggest_priority(threats, client) if USE_LLM else [t["threat_id"] for t in sorted(threats, key=_priority_score, reverse=True)]
                action_payload = {"action_type": "coordinate", "coordination": {"priority_order": order}}
            elif phase == "rescue":
                target = sorted(active_zones, key=lambda z: z.get("total_victims", 0) - z.get("rescued", 0), reverse=True)[0]
                # Send max units to maximize saved_ratio and speed_score
                units = min(5, budget)
                action_payload = {"action_type": "rescue", "rescue": {"zone_id": target["zone_id"], "rescue_units_to_send": units}}

            if not action_payload: break

            # EXECUTE
            res = await http_step(action_payload, session_id)
            reward = res.get("reward", 0.0)
            done = res.get("done", False)
            obs = res.get("observation", {})
            
            # UPDATE STATE
            threats = obs.get("threats", threats)
            resources = obs.get("resources", resources)
            zones = obs.get("affected_zones", zones)
            budget = int(obs.get("resource_budget_remaining", budget))
            rewards.append(reward)
            
            log_step(step, phase, reward, done, None)

            # UPDATE TRACKING
            if phase == "classify": classified.add(target_threat["threat_id"])
            if phase == "predict": predicted.add(target_threat["threat_id"])
            if phase == "allocate" and action_payload: allocated.add(target_threat["threat_id"])
            if phase == "coordinate": coordinated, last_coord_step = True, step

        # FINAL
        state = await http_state(session_id)
        final_score = state.get("final_score", 0.0)
        log_end(done, step, final_score, rewards)

    except Exception as e:
        log_end(False, 0, 0.0, [])
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())
