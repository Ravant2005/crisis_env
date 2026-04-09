from __future__ import annotations
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from server.crisis_env_environment import CrisisEnvEnvironment
from grader import TASK_REGISTRY

# ── Tags for Swagger UI ────────────────────────────────────────────────────────
tags_metadata = [
    {"name": "System", "description": "Health checks and task listing."},
    {"name": "OpenEnv", "description": "Core RL environment endpoints: reset, step, state."},
]

app = FastAPI(
    title="CrisisAI OpenEnv Environment",
    version="1.0.0",
    description="OpenEnv RL environment for AI Crisis Response & Rescue Coordination with multiple difficulty levels.",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

env = CrisisEnvEnvironment()

# ──────────────────────────────────────────────────────────────────────────────
# Difficulty ↔ task_id mapping (used in both directions)
# ──────────────────────────────────────────────────────────────────────────────

_DIFF_TO_TASK: Dict[str, str] = {
    "easy":   "task_easy",
    "medium": "task_medium",
    "hard":   "task_hard",
}

_TASK_TO_DIFF: Dict[str, str] = {
    "task_easy":   "easy",
    "task_medium": "medium",
    "task_hard":   "hard",
}


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic request/response models
# ──────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """
    All fields are optional so the platform can POST with no body (null).
    Accepts task_id (e.g. "task_easy") —
    whichever the caller provides.
    """
    task_id:    Optional[str] = None   # "task_easy" | "task_medium" | "task_hard"
    seed:       Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResult(BaseModel):
    observation: Dict[str, Any]
    reward:      float
    done:        bool
    info:        Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Root redirect
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "1.0.0",
        "tasks_available": sorted(_TASK_TO_DIFF.keys()),
    }


@app.get("/tasks", tags=["System"])
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id":           info["id"],
                "name":         info["name"],
                "difficulty":   info["difficulty"],
                "description":  info["description"],
                "grader_range": info["grader_range"],
                "inbox_size":   info.get("inbox_size", 10),
            }
            for info in TASK_REGISTRY.values()
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# OPENENV endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/reset", tags=["OpenEnv"])
def reset(req: Optional[ResetRequest] = Body(default=None)) -> Dict[str, Any]:
    """
    Reset the environment and return the initial observation as a flat dict.
    Supports empty/null body for OpenEnv platform compliance.
    """
    # If req is None (null body), create a default request
    if req is None:
        req = ResetRequest()

    task_id = req.task_id or "task_easy"
    seed    = req.seed

    result = env.reset(task_id=task_id, seed=seed)

    # Ensure task_id is returned in info for platform validator
    if "info" not in result:
        result["info"] = {}
    result["info"]["task_id"] = task_id

    return {
        "observation": result["observation"],
        "info": result["info"]
    }


@app.post("/step", tags=["OpenEnv"], response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    """Submit one action. Returns observation, reward, done, info."""
    try:
        result = env.step(req.action)
        return {
            "observation": result["observation"],
            "reward": float(result["reward"]),
            "done": bool(result["done"]),
            "info": result.get("info", {})
        }
    except Exception as e:
        # Log the error (visible in server console) but return a valid StepResult
        import traceback
        traceback.print_exc()
        return {
            "observation": {
                "threats": [],
                "resources": [],
                "affected_zones": [],
                "time_remaining": 0,
                "current_step": 0,
                "alerts": [f"Internal error: {str(e)}"],
                "episode_id": "error",
                "resource_budget_remaining": 0,
                "resource_budget_total": 0,
                "recent_actions": [],
                "valid_actions": {"action_mask": [0]*7}
            },
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)}
        }


@app.get("/state", tags=["OpenEnv"])
def state() -> Dict[str, Any]:
    """Return the full internal CrisisState as a flat dict."""
    return env.get_full_state()


@app.get("/scores", tags=["OpenEnv"])
def get_scores():
    """
    Returns all task scores + final score.
    Required for OpenEnv validation + inference script.
    
    CRITICAL: All scores must be STRICTLY between 0 and 1 (not 0.0, not 1.0)
    """
    try:
        scores = env.task_scores()
        
        # Helper to clamp scores to (0, 1) - strictly exclusive!
        def safe_score(val, default=0.5):
            """
            Clamp value to (0.001, 0.999) range.
            Competition requires: 0 < score < 1 (NOT 0.0, NOT 1.0)
            """
            try:
                f = float(val)
                # Force into (0.001, 0.999) range - never exactly 0 or 1
                if f <= 0.0:
                    return 0.001  # Minimum positive value
                elif f >= 1.0:
                    return 0.999  # Maximum value < 1.0
                else:
                    # Ensure we're not too close to boundaries
                    if f < 0.001:
                        return 0.001
                    elif f > 0.999:
                        return 0.999
                    else:
                        return f
            except (ValueError, TypeError):
                return default
        
        # Extract and clamp each score
        classification = safe_score(scores.get("classification"))
        prediction = safe_score(scores.get("prediction"))
        allocation = safe_score(scores.get("allocation"))
        coordination = safe_score(scores.get("coordination"))
        rescue = safe_score(scores.get("rescue"))
        
        # Calculate final score (also clamped)
        raw_final = (
            0.20 * classification +
            0.20 * prediction +
            0.20 * allocation +
            0.15 * coordination +
            0.25 * rescue
        )
        
        # Clamp final score too!
        final_score = safe_score(raw_final, default=0.5)
        
        result = {
            "classification": round(classification, 4),
            "prediction": round(prediction, 4),
            "allocation": round(allocation, 4),
            "coordination": round(coordination, 4),
            "rescue": round(rescue, 4),
            "final": round(final_score, 4),
            "final_score": round(final_score, 4)  # Include both formats
        }
        
        # Debug log (optional - comment out for production)
        # print(f"[DEBUG] /scores returning: {result}", flush=True)
        
        return result
    
    except Exception as e:
        # NEVER return invalid JSON - always return safe defaults
        print(f"[ERROR] /scores failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        # Safe fallback values (all strictly between 0 and 1)
        return {
            "classification": 0.5000,
            "prediction": 0.5000,
            "allocation": 0.5000,
            "coordination": 0.5000,
            "rescue": 0.5000,
            "final": 0.5000,
            "final_score": 0.5000,
            "error": str(e)
        }


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (mandatory for platform validator)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()