"""
inference.py — Deterministic baseline agent for the AI Crisis Response & Rescue Coordination Environment.

Strategy:
    priority_score = (severity * population_at_risk) / max(time_to_impact, 1)

Pipeline per episode:
    1. CLASSIFY  — identify all active threats
    2. PREDICT   — estimate TTI + population for each threat
    3. COORDINATE — rank threats by priority score
    4. ALLOCATE  — assign best available resource per threat (zone-affinity aware)
    5. RESCUE    — deploy units into all impacted zones every step

Logging format (mandatory):
    [START]
    [STEP N] action_type | threat/zone | result | reward | done
    ...
    [END]
    [SCORE] task_scores + final

Usage:
    python3 inference.py                          # connects to localhost:8000
    API_BASE_URL=http://... python3 inference.py  # remote server
    SEED=42 python3 inference.py                  # fixed seed for reproducibility
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import requests
import websocket   # websocket-client
from typing import Any, Dict, List, Optional, Tuple

# ─── Optional: LLM client (OpenAI-compatible) ───────────────────────────────
# Used for agentic enhancement; falls back to rule-based if unavailable.
try:
    from openai import OpenAI
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIGURATION  (read from environment)
# ─────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
WS_URL:       str = API_BASE_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")
SEED:         int = int(os.environ.get("SEED", "42"))
USE_LLM:      bool = os.environ.get("USE_LLM", "false").lower() == "true" and _LLM_AVAILABLE
MAX_RETRIES:  int = 3
STEP_DELAY:   float = 0.02   # seconds between steps (faster processing)
MAX_ACTIONS_PER_STEP = 6    # execute up to 6 actions per step (increased to fit coordinate + allocate)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

_step_counter: int = 0
_cumulative_score: float = 0.0

def log_start():
    print("[START]", flush=True)

def log_step(action_type: str, target: str, result: str, reward: float, done: bool, decision_reasoning: str = "", cumulative_score: float = 0.0):
    global _step_counter, _cumulative_score
    _step_counter += 1
    if cumulative_score is not None:
        _cumulative_score = cumulative_score
    else:
        _cumulative_score = 0.0
    
    reasoning_str = f" | reasoning={decision_reasoning}" if decision_reasoning else ""
    print(
        f"[STEP {_step_counter}] "
        f"action={action_type} | target={target} | result={result} | "
        f"reward={reward:.4f} | done={done}{reasoning_str} | cumulative_score={_cumulative_score:.4f}",
        flush=True,
    )

def log_end():
    print("[END]", flush=True)

def log_score(scores: Dict[str, Any]):
    print(
        f"[SCORE] "
        f"classification={scores.get('classification', 0):.4f} | "
        f"prediction={scores.get('prediction', 0):.4f} | "
        f"allocation={scores.get('allocation', 0):.4f} | "
        f"coordination={scores.get('coordination', 0):.4f} | "
        f"rescue={scores.get('rescue', 0):.4f} | "
        f"final={scores.get('final', 0):.4f}",
        flush=True,
    )

def log_info(msg: str):
    print(f"[INFO] {msg}", flush=True)

def log_error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ─────────────────────────────────────────────
# HTTP CLIENT
# ─────────────────────────────────────────────

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


def http_reset(seed: int = SEED, difficulty: str = "medium", session_id: str = "test_session") -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{API_BASE_URL}/reset",
                json={"seed": seed, "difficulty": difficulty, "session_id": session_id},
                headers=_headers(),
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log_error(f"reset attempt {attempt+1} failed: {exc}")
            time.sleep(1)
    raise RuntimeError("Failed to reset environment after max retries.")


def http_step(action: Dict[str, Any], session_id: str = "test_session") -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": action, "session_id": session_id},
                headers=_headers(),
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log_error(f"step attempt {attempt+1} failed: {exc}")
            time.sleep(1)
    raise RuntimeError("Failed to execute step after max retries.")


def http_state(session_id: str = "test_session") -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                f"{API_BASE_URL}/state",
                params={"session_id": session_id},
                headers=_headers(),
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log_error(f"state attempt {attempt+1} failed: {exc}")
            time.sleep(1)
    raise RuntimeError("Failed to fetch state after max retries.")

# ─────────────────────────────────────────────
# RULE-BASED DECISION ENGINE
# ─────────────────────────────────────────────

# Zone → preferred resource types (mirrors environment constants)
_ZONE_AFFINITY: Dict[str, List[str]] = {
    "military": ["military_unit", "medical_team"],
    "maritime": ["coast_guard", "rescue_drone"],
    "urban":    ["swat_team", "fire_brigade", "evacuation_bus"],
    "rural":    ["fire_brigade", "medical_team", "rescue_drone"],
}


def _preferred_action(threat: Dict[str, Any]) -> str:
    """Determine preferred action based on threat characteristics.
    Uses recommended_action_hint from observation if available, otherwise computes."""
    hint = threat.get("recommended_action_hint")
    if hint:
        return hint
    
    tti = max(int(threat.get("time_to_impact", 1)), 1)
    pop = int(threat.get("population_at_risk", 0))
    sev = float(threat.get("severity", 1.0))
    
    if tti <= 2 and pop > 1000:
        return "evacuate"
    elif sev >= 4:
        return "allocate_resources"
    else:
        return "classify_and_monitor"


def _priority_score(threat: Dict[str, Any]) -> float:
    """Core priority formula: severity × population / max(TTI, 1)."""
    sev = float(threat.get("severity", 1.0))
    pop = int(threat.get("population_at_risk", 1))
    tti = max(int(threat.get("time_to_impact", 1)), 1)
    return (sev * pop) / tti


def _best_resource(
    threat: Dict[str, Any],
    resources: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select the best available resource for a threat.
    Priority: zone-affinity match → highest effectiveness → any available.
    """
    zone        = threat.get("zone", "")
    preferred   = _ZONE_AFFINITY.get(zone, [])
    available   = [r for r in resources if r.get("is_available", False)]

    if not available:
        return None

    # Score each resource
    def resource_score(r: Dict[str, Any]) -> float:
        base  = float(r.get("effectiveness", 0.5))
        bonus = 0.3 if r.get("resource_type") in preferred else 0.0
        return base + bonus

    return max(available, key=resource_score)


def _classify_action(threat: Dict[str, Any]) -> Dict[str, Any]:
    """Build a CLASSIFY action for a given threat."""
    return {
        "action_type":    "classify",
        "classification": {
            "threat_id":          threat["threat_id"],
            "predicted_type":     threat["threat_type"],    # rule: trust observation
            "predicted_severity": threat["severity"],       # rule: trust observation
        },
    }


def _predict_action(threat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a PREDICT action.
    Uses observed values directly — minimises grader error.
    """
    tti      = max(int(threat.get("time_to_impact", 5)), 1)
    pop      = int(threat.get("population_at_risk", 100))
    severity = float(threat.get("severity", 5.0))

    # Use observed values directly — minimises grader error
    estimated_pop = pop

    return {
        "action_type": "predict",
        "prediction": {
            "threat_id":     threat["threat_id"],
            "predicted_tti": tti,
            "predicted_pop": estimated_pop,
        },
    }


def _coordinate_action(threats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a COORDINATE action — rank active threats by priority score."""
    active = [t for t in threats if t.get("status") == "active"]
    ranked = sorted(active, key=_priority_score, reverse=True)
    return {
        "action_type":  "coordinate",
        "coordination": {
            "priority_order": [t["threat_id"] for t in ranked],
        },
    }


def _allocate_action(
    threat: Dict[str, Any],
    resources: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build an ALLOCATE action pairing a threat with its best available resource."""
    res = _best_resource(threat, resources)
    if res is None:
        return None
    return {
        "action_type": "allocate",
        "allocation":  {
            "threat_id":   threat["threat_id"],
            "resource_id": res["resource_id"],
        },
    }


def _rescue_action(zone: Dict[str, Any], budget_remaining: int = 99) -> Optional[Dict[str, Any]]:
    """
    Build a RESCUE action - send more units for better rescue score.
    """
    remaining = zone.get("total_victims", 0) - zone.get("rescued", 0)
    if remaining <= 0:
        return None

    # Send more units - up to 2-3 at a time for better rescue score
    if remaining > 300:
        units = 3
    elif remaining > 150:
        units = 2
    else:
        units = 1

    return {
        "action_type": "rescue",
        "rescue": {
            "zone_id":              zone["zone_id"],
            "rescue_units_to_send": units,
        },
    }

# ─────────────────────────────────────────────
# OPTIONAL: LLM-ASSISTED DECISION LAYER
# ─────────────────────────────────────────────

def _llm_suggest_priority(
    threats: List[Dict[str, Any]],
    client: Any,
) -> List[int]:
    """
    Ask the LLM to rank threat IDs by priority.
    Returns a list of threat_ids in priority order.
    Falls back to rule-based if LLM fails.
    """
    active = [t for t in threats if t.get("status") == "active"]
    if not active:
        return []

    prompt = (
        "You are a crisis coordinator. Rank the following threats from highest to lowest priority.\n"
        "Priority formula: severity × population_at_risk / time_to_impact\n\n"
        "Threats:\n"
        + "\n".join(
            f"- threat_id={t['threat_id']}, type={t['threat_type']}, "
            f"severity={t['severity']}, population={t['population_at_risk']}, "
            f"tti={t['time_to_impact']}, zone={t['zone']}"
            for t in active
        )
        + "\n\nRespond ONLY with a JSON array of threat_ids in priority order. Example: [3,1,2]"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,
        )
        raw  = response.choices[0].message.content.strip()
        ids  = json.loads(raw)
        if isinstance(ids, list) and all(isinstance(x, int) for x in ids):
            return ids
    except Exception as exc:
        log_error(f"LLM suggestion failed: {exc}. Falling back to rule-based.")

    # Fallback
    return [t["threat_id"] for t in sorted(active, key=_priority_score, reverse=True)]


# ─────────────────────────────────────────────
# MAIN EPISODE LOOP
# ─────────────────────────────────────────────


def run_episode(seed: int = SEED, difficulty: str = "medium") -> Dict[str, float]:
    """
    Run one full episode of the Crisis Response environment.
    Returns the final grader scores.
    """
    global _step_counter, _cumulative_score
    _step_counter = 0
    _cumulative_score = 0.0
    
    # Use a unique session ID for this episode to avoid state conflicts
    import uuid
    session_id = f"episode_{uuid.uuid4().hex[:8]}"
    
    log_info(f"Connecting to {API_BASE_URL} | seed={seed} | difficulty={difficulty} | session={session_id} | use_llm={USE_LLM}")

    # ── Optional LLM client ────────────────────────────────────────────────
    llm_client = None
    if USE_LLM and _LLM_AVAILABLE:
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        hf_token = os.getenv("HF_TOKEN", None)

        # For HuggingFace inference endpoints, use /v1 suffix for OpenAI compatibility
        # For localhost, don't set base_url (not needed for local inference)
        if base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1"):
            # Local server - don't use OpenAI client for LLM calls
            openai_base_url = None
        else:
            # Remote server (HuggingFace, etc.) - use OpenAI-compatible endpoint
            openai_base_url = base_url.rstrip("/") + "/v1"

        llm_client = OpenAI(
            base_url=openai_base_url,
            api_key=hf_token if hf_token else "EMPTY"
        )
        log_info(f"LLM client initialised — model={model_name} base_url={openai_base_url}")

    # ── Reset ──────────────────────────────────────────────────────────────
    reset_resp  = http_reset(seed=seed, difficulty=difficulty, session_id=session_id)
    observation = reset_resp.get("observation", {})
    threats     = observation.get("threats", [])
    resources   = observation.get("resources", [])
    zones       = observation.get("affected_zones", [])
    done        = False

    log_info(
        f"Episode started — "
        f"{len(threats)} threats | {len(resources)} resources | "
        f"time_remaining={observation.get('time_remaining', 0)}"
    )

    # ── Track which tasks have been done this episode ──────────────────────
    classified:  set = set()
    predicted:   set = set()
    coordinated: bool = False
    allocated:   set = set()

    # ── Phase completion flags ─────────────────────────────────────────────
    classify_phase_done = False
    predict_phase_done = False
    coordinate_phase_done = False
    
    # Track when phases were completed
    classify_completed_step = 0
    predict_completed_step = 0

    # ── Episode loop ───────────────────────────────────────────────────────
    step = 0
    _live_resource_budget = 99  # initialized; updated after every http_step call
    while not done and step < 50:
        step += 1
        
        # Get current active threats and zones
        active_threats = [t for t in threats if t.get("status") == "active"]
        impacted_zones = [z for z in zones if z.get("is_active", False)]
        
        # Collect actions to execute this step (max MAX_ACTIONS_PER_STEP)
        actions_to_execute: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        
        # Track whether coordinate runs this step (to skip allocation in same step)
        coordinate_this_step = False
        

        if not classify_phase_done:
            classify_actions = []
            # ALWAYS re-derive active threats from the latest observation — never use
            # a stale list from reset time. Threat IDs change as threats impact/spawn.
            current_active = [t for t in threats if t.get("status") == "active"]
            valid_ids = {t["threat_id"] for t in current_active}
            # Prune classified set — remove IDs that are no longer active/valid
            classified = classified & valid_ids
            for threat in current_active:
                tid = threat["threat_id"]
                if tid not in classified:
                    classify_actions.append(("classify", threat, _classify_action(threat)))
                    classified.add(tid)
            actions_to_execute.extend(classify_actions)
            if current_active and all(t["threat_id"] in classified for t in current_active):
                classify_phase_done = True
                classify_completed_step = step
        
        # ── Phase 2: PREDICT all active threats (if not done) ─────────────
        if not predict_phase_done:
            predict_actions = []
            current_active = [t for t in threats if t.get("status") == "active"]
            valid_ids = {t["threat_id"] for t in current_active}
            # Prune predicted set — remove IDs that are no longer active/valid
            predicted = predicted & valid_ids
            for threat in current_active:
                tid = threat["threat_id"]
                if tid not in predicted:
                    predict_actions.append(("predict", threat, _predict_action(threat)))
                    predicted.add(tid)
            actions_to_execute.extend(predict_actions)
            if current_active and all(t["threat_id"] in predicted for t in current_active):
                predict_phase_done = True
                predict_completed_step = step
        
        # ── Phase 3: COORDINATE (once when classification + prediction done) ─
        # Add coordinate first, then add allocation actions
        # Only run coordinate if both classify and predict were completed in PREVIOUS steps
        if not coordinate_phase_done and classify_phase_done and predict_phase_done:
            if active_threats and step > classify_completed_step and step > predict_completed_step:
                if llm_client:
                    order = _llm_suggest_priority(active_threats, llm_client)
                    coord_action = {
                        "action_type":  "coordinate",
                        "coordination": {"priority_order": order},
                    }
                else:
                    # Pass ONLY currently active threats — not the full stale list
                    coord_action = _coordinate_action(
                        [t for t in threats if t.get("status") == "active"]
                    )
                actions_to_execute.append(("coordinate", {"threat_id": "all"}, coord_action))
                coordinated = True
                coordinate_this_step = True
                # Don't set coordinate_phase_done here - set after execution
        
        # ── Phase 4: ALLOCATE resources to unallocated threats ────────────
        # Always attempt allocation — do NOT skip just because coordinate ran.
        # Threats have short TTIs; waiting one more step means they impact unmitigated.
        current_active_threats = [t for t in threats if t.get("status") == "active"]
        ranked_threats = sorted(current_active_threats, key=_priority_score, reverse=True)
        # Use the live budget tracked from observation updates
        budget_remaining = _live_resource_budget if "_live_resource_budget" in dir() else 99
        allocation_actions = []
        for threat in ranked_threats:
            tid = threat["threat_id"]
            if tid not in allocated and threat.get("assigned_resource") is None:
                alloc = _allocate_action(threat, resources)
                if alloc:
                    allocation_actions.append(("allocate", threat, alloc))
                    for r in resources:
                        if r["resource_id"] == alloc["allocation"]["resource_id"]:
                            r["is_available"] = False
        actions_to_execute.extend(allocation_actions)
        
        # ── Phase 5: RESCUE impacted zones ─────────────────────────────────
        current_impacted_zones = [z for z in zones if z.get("is_active", False)]
        sorted_zones = sorted(
            current_impacted_zones,
            key=lambda z: (z.get("total_victims", 0) - z.get("rescued", 0)),
            reverse=True
        )
        _live_budget = _live_resource_budget if "_live_resource_budget" in dir() else (
            int(threats[0].get("resource_budget_remaining", 99)) if threats else 99
        )
        rescue_budget_per_zone = max(1, _live_budget // max(len(sorted_zones), 1))
        for zone in sorted_zones:
            rescue_act = _rescue_action(zone, budget_remaining=_live_budget)
            if rescue_act:
                actions_to_execute.append(("rescue", zone, rescue_act))
                # Reduce estimated budget so next zone gets a fair share
                _live_budget = max(0, _live_budget - rescue_act["rescue"]["rescue_units_to_send"])
        
        # ── Execute collected actions (up to MAX_ACTIONS_PER_STEP) ─────────
        executed = 0
        
        for action_label, target_obj, action_payload in actions_to_execute[:MAX_ACTIONS_PER_STEP]:
            executed += 1
            
            target_label = (
                f"threat_{target_obj.get('threat_id', '?')}"
                if "threat_id" in target_obj
                else f"zone_{target_obj.get('zone_id', '?')}"
            )
            
            # Determine decision reasoning
            if action_label == "classify":
                reasoning = "classify threat to understand its characteristics"
            elif action_label == "predict":
                reasoning = "predict TTI and population for accurate resource allocation"
            elif action_label == "coordinate":
                reasoning = "set priority order for all active threats"
            elif action_label == "allocate":
                pref = _preferred_action(target_obj)
                reasoning = f"assign best resource to high-priority threat ({pref})"
            elif action_label == "rescue":
                reasoning = "deploy rescue units to save victims in impacted zone"
            else:
                reasoning = "taking action on critical threat"
            
            # Execute action with session_id
            result = http_step(action_payload, session_id)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs_new = result.get("observation", {})
            alerts = obs_new.get("alerts", [])
            
            # Update local state from fresh observation
            threats    = obs_new.get("threats",        threats)
            resources  = obs_new.get("resources",      resources)
            zones      = obs_new.get("affected_zones", zones)
            # Track live budget — used by rescue and allocation sizing
            _live_resource_budget = int(obs_new.get("resource_budget_remaining", 99))
            
            # Get cumulative score with session_id
            state = http_state(session_id)
            cumulative = state.get("final_score", 0.0)
            
            result_label = alerts[0][:80] if alerts else "ok"
            log_step(action_label, target_label, result_label, reward, done, reasoning, cumulative)
            
            if done:
                break
        
        # After executing actions, update tracking sets for successfully completed actions
        for action_label, target_obj, action_payload in actions_to_execute[:executed]:
            if action_label == "allocate":
                alloc_tid = action_payload.get("allocation", {}).get("threat_id")
                if alloc_tid:
                    allocated.add(alloc_tid)
        
        # If coordinate ran this step, mark as done
        if coordinate_this_step:
            coordinate_phase_done = True
        
        # After executing actions, check if we need to continue phases
        # Re-check phases in case new threats appeared or tasks weren't complete
        current_active = [t for t in threats if t.get("status") == "active"]
        
        # If there are still unclassified threats, reset the flag
        unclassified = [t for t in current_active if t["threat_id"] not in classified]
        if unclassified:
            classify_phase_done = False
            
        # If there are still unpredicted threats, reset the flag  
        unpredicted = [t for t in current_active if t["threat_id"] not in predicted]
        if unpredicted:
            predict_phase_done = False
        
        # If classification or prediction was reset, also reset coordinate
        if not classify_phase_done or not predict_phase_done:
            coordinate_phase_done = False
        
        # Update global active_threats for next iteration
        active_threats = current_active
        
        # Small delay between steps
        time.sleep(STEP_DELAY)
        
        # Check if episode is done
        if done:
            break
        
        # Check if we need to re-coordinate (new threats appeared, not just allocation pending)
        # Only re-run if new UNCLASSIFIED or UNPREDICTED threats appeared
        current_threats = [t for t in threats if t.get("status") == "active"]
        new_threats_appeared = any(t["threat_id"] not in classified or t["threat_id"] not in predicted for t in current_threats)
        if new_threats_appeared:
            classify_phase_done = False
            predict_phase_done = False
            coordinate_phase_done = False

    # ── Final scores ───────────────────────────────────────────────────────
    state = http_state(session_id)
    scores = {
        "classification": state.get("classification_score", 0.0),
        "prediction":     state.get("prediction_score", 0.0),
        "allocation":     state.get("allocation_score", 0.0),
        "coordination":   state.get("coordination_score", 0.0),
        "rescue":         state.get("rescue_score", 0.0),
        "final":          state.get("final_score", 0.0),
    }

    log_end()
    log_score(scores)
    return scores


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log_start()

    try:
        scores = run_episode(seed=SEED, difficulty="medium")
        log_info(f"Run complete — final_score={scores['final']:.4f}")
        sys.exit(0)

    except KeyboardInterrupt:
        log_error("Interrupted by user.")
        sys.exit(1)

    except Exception as exc:
        import traceback
        log_error(f"Fatal: {exc}")
        traceback.print_exc()
        log_end()
        sys.exit(2)