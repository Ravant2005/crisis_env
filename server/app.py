"""
server/app.py — FastAPI server for the AI Crisis Response & Rescue Coordination Environment.
Implements the full OpenEnv server protocol:
  REST:      GET /  |  GET /health  |  GET /tasks  |  GET /state  |  GET /scores
             POST /reset  |  POST /step
  WebSocket: /ws  (primary agentic interface)
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    CrisisAction, CrisisObservation, CrisisState, StepResult,
    ActionType,
)
from server.environment import CrisisEnvironment

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI Crisis Response & Rescue Coordination — OpenEnv",
    description=(
        "A real-world OpenEnv environment where an AI agent classifies threats, "
        "predicts impact, allocates resources, coordinates multi-threat response, "
        "and optimizes rescue operations to maximise lives saved."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# GLOBAL ENVIRONMENT REGISTRY
# ─────────────────────────────────────────────

_environments: Dict[str, CrisisEnvironment] = {}
_DEFAULT_SESSION = "default"


def _get_env(session_id: str = _DEFAULT_SESSION) -> CrisisEnvironment:
    if session_id not in _environments:
        _environments[session_id] = CrisisEnvironment()
    return _environments[session_id]


def _new_env(session_id: str = _DEFAULT_SESSION, seed: Optional[int] = None) -> CrisisEnvironment:
    env = CrisisEnvironment(seed=seed)
    _environments[session_id] = env
    return env


# ─────────────────────────────────────────────
# SERIALISATION HELPERS
# ─────────────────────────────────────────────

def _obs_to_dict(obs: CrisisObservation) -> Dict[str, Any]:
    return obs.model_dump()


def _state_to_dict(state: CrisisState) -> Dict[str, Any]:
    return state.model_dump()


def _step_to_dict(result: StepResult) -> Dict[str, Any]:
    return {
        "observation": _obs_to_dict(result.observation),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


def _error_response(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "code": code, "message": message}


# ─────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """Root endpoint — required for Hugging Face Space validation."""
    return {
        "name":        "CrisisAI: AI Crisis Response & Rescue Coordination",
        "version":     "1.0.0",
        "status":      "running",
        "description": "OpenEnv-compliant RL environment for multi-threat emergency coordination.",
        "endpoints": {
            "health": "GET  /health",
            "reset":  "POST /reset",
            "step":   "POST /step",
            "state":  "GET  /state",
            "scores": "GET  /scores",
            "tasks":  "GET  /tasks",
            "ws":     "WS   /ws",
            "docs":   "GET  /docs",
        },
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe — required for HuggingFace Space automated validation."""
    return {"status": "ok", "service": "crisis-response-openenv", "version": "1.0.0"}


@app.get("/tasks", tags=["OpenEnv"])
async def list_tasks():
    """Return all task definitions with grader metadata."""
    return {
        "tasks": [
            {
                "task_id":     1,
                "name":        "Threat Classification",
                "difficulty":  "easy",
                "action_type": ActionType.CLASSIFY,
                "grader":      "classification_score",
                "grader_range": [0.0, 1.0],
                "score_range": [0.0, 1.0],
                "description": "Identify the type and severity of each active threat. Score = 0.70 × correct_ratio + 0.30 × severity_accuracy.",
            },
            {
                "task_id":     2,
                "name":        "Impact Prediction",
                "difficulty":  "medium",
                "action_type": ActionType.PREDICT,
                "grader":      "prediction_score",
                "grader_range": [0.0, 1.0],
                "score_range": [0.0, 1.0],
                "description": "Predict time-to-impact and population affected. Score = 1.0 - (0.5 × tti_err + 0.5 × pop_err).",
            },
            {
                "task_id":     3,
                "name":        "Resource Allocation",
                "difficulty":  "medium",
                "action_type": ActionType.ALLOCATE,
                "grader":      "allocation_score",
                "grader_range": [0.0, 1.0],
                "score_range": [0.0, 1.0],
                "description": "Assign zone-matched resources to threats. Score = 0.45 × effectiveness + 0.30 × zone_affinity + 0.15 × budget_efficiency.",
            },
            {
                "task_id":     4,
                "name":        "Multi-Threat Coordination",
                "difficulty":  "hard",
                "action_type": ActionType.COORDINATE,
                "grader":      "coordination_score",
                "grader_range": [0.0, 1.0],
                "score_range": [0.0, 1.0],
                "description": "Rank threats by priority. Score = rank_correlation(agent_order, ideal_order).",
            },
            {
                "task_id":     5,
                "name":        "Rescue Optimisation",
                "difficulty":  "hard",
                "action_type": ActionType.RESCUE,
                "grader":      "rescue_score",
                "grader_range": [0.0, 1.0],
                "score_range": [0.0, 1.0],
                "description": "Deploy rescue units to save victims. Score = 0.65 × saved_ratio + 0.20 × speed_score + 0.15 × resource_efficiency.",
            },
        ]
    }


@app.post("/reset", tags=["OpenEnv"])
async def reset_endpoint(body: Optional[Dict[str, Any]] = None):
    """
    Reset the environment and return the initial observation.
    Optional body: { "seed": <int>, "difficulty": <str>, "session_id": <str> }
    """
    body       = body or {}
    seed       = body.get("seed", None)
    difficulty = body.get("difficulty", "medium")
    session_id = body.get("session_id", _DEFAULT_SESSION)

    env = _new_env(session_id=session_id, seed=seed)
    obs = env.reset(seed=seed, difficulty=difficulty)
    return {"status": "ok", "observation": _obs_to_dict(obs)}


@app.post("/step", tags=["OpenEnv"])
async def step_endpoint(body: Dict[str, Any]):
    """
    Submit one action and advance the simulation by one step.
    Body: { "action": <CrisisAction>, "session_id": <str> (optional) }
    Returns: { observation, reward, done, info }
    """
    session_id  = body.get("session_id", _DEFAULT_SESSION)
    env         = _get_env(session_id)
    action_data = body.get("action")

    if action_data is None:
        raise HTTPException(status_code=422, detail="Missing 'action' field in request body.")

    try:
        action = CrisisAction(**action_data)
    except (ValidationError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    result = env.step(action)
    return _step_to_dict(result)


@app.get("/state", tags=["OpenEnv"])
async def state_endpoint(session_id: str = _DEFAULT_SESSION):
    """Return the current episode state and all task grader scores."""
    env   = _get_env(session_id)
    state = env.state()
    return _state_to_dict(state)


@app.get("/scores", tags=["OpenEnv"])
async def scores_endpoint(session_id: str = _DEFAULT_SESSION):
    """Convenience endpoint returning only the grader scores for all 5 tasks."""
    env   = _get_env(session_id)
    state = env.state()
    return {
        "classification": round(state.classification_score, 4),
        "prediction":     round(state.prediction_score,     4),
        "allocation":     round(state.allocation_score,     4),
        "coordination":   round(state.coordination_score,   4),
        "rescue":         round(state.rescue_score,         4),
        "final":          round(state.final_score,          4),
        "final_score":    round(state.final_score,          4),
        "episode_id":     state.episode_id,
        "done":           state.done,
    }


# ─────────────────────────────────────────────
# WEBSOCKET — PRIMARY AGENTIC INTERFACE  (/ws)
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Primary WebSocket interface for agentic interaction."""
    await websocket.accept()

    session_id = f"ws_{id(websocket)}"
    env        = _new_env(session_id=session_id)

    async def send(msg_type: str, data: Any = None):
        payload: Dict[str, Any] = {"type": msg_type}
        if data is not None:
            payload["data"] = data
        await websocket.send_text(json.dumps(payload, default=str))

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send("error", _error_response("PARSE_ERROR", "Invalid JSON received."))
                continue

            command = msg.get("command", "").strip().lower()

            if command == "ping":
                await send("pong")

            elif command == "reset":
                seed = msg.get("seed", None)
                env  = _new_env(session_id=session_id, seed=seed)
                obs  = env.reset()
                await send("observation", _obs_to_dict(obs))

            elif command == "step":
                action_data = msg.get("action")
                if not action_data:
                    await send("error", _error_response(
                        "MISSING_ACTION", "Field 'action' is required for step command."
                    ))
                    continue

                try:
                    action = CrisisAction(**action_data)
                except (ValidationError, Exception) as exc:
                    await send("error", _error_response("INVALID_ACTION", str(exc)))
                    continue

                try:
                    result = env.step(action)
                    await send("step_result", _step_to_dict(result))
                    if result.done:
                        await send("state", _state_to_dict(env.state()))
                except Exception as exc:
                    await send("error", _error_response(
                        "STEP_ERROR", f"Simulation error: {exc}\n{traceback.format_exc()}"
                    ))

            elif command == "state":
                await send("state", _state_to_dict(env.state()))

            elif command == "tasks":
                tasks_resp = await list_tasks()
                await send("tasks", tasks_resp)

            elif command == "scores":
                task_scores = env.task_scores()
                st = env.state()
                await send("scores", {
                    "classification": round(task_scores.get("classification", 0.0), 4),
                    "prediction":     round(task_scores.get("prediction",     0.0), 4),
                    "allocation":     round(task_scores.get("allocation",     0.0), 4),
                    "coordination":   round(task_scores.get("coordination",   0.0), 4),
                    "rescue":         round(task_scores.get("rescue",         0.0), 4),
                    "final":          round(st.final_score, 4),
                })

            else:
                await send("error", _error_response(
                    "UNKNOWN_COMMAND",
                    f"Unknown command '{command}'. Valid: reset, step, state, tasks, scores, ping."
                ))

    except WebSocketDisconnect:
        _environments.pop(session_id, None)

    except Exception as exc:
        try:
            await send("error", _error_response("FATAL_ERROR", f"Unexpected server error: {exc}"))
        except Exception:
            pass
        finally:
            _environments.pop(session_id, None)


# ─────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "code": exc.status_code, "message": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "code": "INTERNAL_ERROR", "message": str(exc)},
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
