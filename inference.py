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
STEP_DELAY:   float = 0.05   # seconds between steps (avoids hammering server)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

_step_counter: int = 0

def log_start():
    print("[START]", flush=True)

def log_step(action_type: str, target: str, result: str, reward: float, done: bool):
    global _step_counter
    _step_counter += 1
    print(
        f"[STEP {_step_counter}] "
        f"action={action_type} | target={target} | result={result} | "
        f"reward={reward:.4f} | done={done}",
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


def http_reset(seed: int = SEED) -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{API_BASE_URL}/reset",
                json={"seed": seed},
                headers=_headers(),
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log_error(f"reset attempt {attempt+1} failed: {exc}")
            time.sleep(1)
    raise RuntimeError("Failed to reset environment after max retries.")


def http_step(action: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": action},
                headers=_headers(),
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log_error(f"step attempt {attempt+1} failed: {exc}")
            time.sleep(1)
    raise RuntimeError("Failed to execute step after max retries.")


def http_state() -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                f"{API_BASE_URL}/state",
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
    Rule: use observed TTI directly; estimate population from severity.
    """
    tti      = max(int(threat.get("time_to_impact", 5)), 1)
    pop      = int(threat.get("population_at_risk", 100))
    severity = float(threat.get("severity", 5.0))

    # Slightly adjust population estimate based on severity (realistic heuristic)
    estimated_pop = int(pop * (0.8 + severity / 50.0))

    return {
        "action_type": "predict",
        "prediction":  {
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


def _rescue_actions(zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build RESCUE actions for all active affected zones."""
    actions = []
    for zone in zones:
        if not zone.get("is_active", False):
            continue
        remaining = zone.get("total_victims", 0) - zone.get("rescued", 0)
        if remaining <= 0:
            continue
        # Send maximum units per zone
        actions.append({
            "action_type": "rescue",
            "rescue": {
                "zone_id":              zone["zone_id"],
                "rescue_units_to_send": 5,
            },
        })
    return actions

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

def run_episode(seed: int = SEED) -> Dict[str, float]:
    """
    Run one full episode of the Crisis Response environment.
    Returns the final grader scores.
    """
    log_info(f"Connecting to {API_BASE_URL} | seed={seed} | use_llm={USE_LLM}")

    # ── Optional LLM client ────────────────────────────────────────────────
    llm_client = None
    if USE_LLM and _LLM_AVAILABLE:
        llm_client = OpenAI(
            base_url=API_BASE_URL if API_BASE_URL != "http://localhost:8000" else None,
            api_key=HF_TOKEN or "sk-placeholder",
        )
        log_info(f"LLM client initialised — model={MODEL_NAME}")

    # ── Reset ──────────────────────────────────────────────────────────────
    reset_resp  = http_reset(seed=seed)
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
    _classified:  set = set()
    _predicted:   set = set()
    _coordinated: bool = False
    _allocated:   set = set()

    # ── Episode loop ───────────────────────────────────────────────────────
    while not done:
        actions_this_step: List[Dict[str, Any]] = []

        active_threats = [t for t in threats if t.get("status") == "active"]
        impacted_zones = [z for z in zones if z.get("is_active", False)]

        # ── Phase 1: Classify any unclassified active threats ────────────
        for threat in active_threats:
            tid = threat["threat_id"]
            if tid not in _classified:
                actions_this_step.append(("classify", threat, _classify_action(threat)))
                _classified.add(tid)

        # ── Phase 2: Predict impact for unclassified threats ─────────────
        for threat in active_threats:
            tid = threat["threat_id"]
            if tid not in _predicted:
                actions_this_step.append(("predict", threat, _predict_action(threat)))
                _predicted.add(tid)

        # ── Phase 3: Coordinate (once per episode, re-run if new threats) ─
        if active_threats and not _coordinated:
            if llm_client:
                order = _llm_suggest_priority(active_threats, llm_client)
                coord_action = {
                    "action_type":  "coordinate",
                    "coordination": {"priority_order": order},
                }
            else:
                coord_action = _coordinate_action(threats)
            actions_this_step.append(("coordinate", {"threat_id": "all"}, coord_action))
            _coordinated = True

        # ── Phase 4: Allocate resources to unallocated high-priority threats
        ranked = sorted(active_threats, key=_priority_score, reverse=True)
        for threat in ranked:
            tid = threat["threat_id"]
            if tid not in _allocated and threat.get("assigned_resource") is None:
                alloc = _allocate_action(threat, resources)
                if alloc:
                    actions_this_step.append(("allocate", threat, alloc))
                    _allocated.add(tid)
                    # Mark resource as used locally to avoid double-assignment
                    for r in resources:
                        if r["resource_id"] == alloc["allocation"]["resource_id"]:
                            r["is_available"] = False

        # ── Phase 5: Rescue all active zones ─────────────────────────────
        rescue_acts = _rescue_actions(impacted_zones)
        for ra in rescue_acts:
            actions_this_step.append(("rescue", {"zone_id": ra["rescue"]["zone_id"]}, ra))

        # ── Execute actions (one per step, pick highest priority) ─────────
        if actions_this_step:
            action_type_label, target_obj, action_payload = actions_this_step[0]

            target_label = (
                f"threat_{target_obj.get('threat_id', '?')}"
                if "threat_id" in target_obj
                else f"zone_{target_obj.get('zone_id', '?')}"
            )

            result     = http_step(action_payload)
            reward     = result.get("reward", 0.0)
            done       = result.get("done", False)
            obs_new    = result.get("observation", {})
            alerts     = obs_new.get("alerts", [])

            # Update local state
            threats   = obs_new.get("threats", threats)
            resources = obs_new.get("resources", resources)
            zones     = obs_new.get("affected_zones", zones)

            result_label = alerts[0][:80] if alerts else "ok"
            log_step(action_type_label, target_label, result_label, reward, done)

            # Re-allow re-coordination if new threats appear
            new_active_ids = {t["threat_id"] for t in threats if t.get("status") == "active"}
            if new_active_ids - _allocated:
                _coordinated = False

        else:
            # Nothing to do this step — send a no-op rescue or wait
            time_remaining = observation.get("time_remaining", 0)
            if time_remaining <= 0:
                break

            # Re-issue coordination as a low-cost action to keep stepping
            if active_threats:
                action_payload = _coordinate_action(threats)
                result  = http_step(action_payload)
                reward  = result.get("reward", 0.0)
                done    = result.get("done", False)
                obs_new = result.get("observation", {})
                threats   = obs_new.get("threats", threats)
                resources = obs_new.get("resources", resources)
                zones     = obs_new.get("affected_zones", zones)
                log_step("coordinate", "rebalance", "priority-refresh", reward, done)
            else:
                # All threats resolved — fetch state and exit
                break

        time.sleep(STEP_DELAY)

    # ── Final scores ───────────────────────────────────────────────────────
    state = http_state()
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
        scores = run_episode(seed=SEED)
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
