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
    Accepts BOTH task_id (e.g. "task_easy") AND difficulty (e.g. "easy") —
    whichever the caller provides.
    """
    task_id:    Optional[str] = None   # "task_easy" | "task_medium" | "task_hard"
    difficulty: Optional[str] = None   # "easy" | "medium" | "hard"
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

    task_id    = req.task_id
    seed       = req.seed
    difficulty = req.difficulty

    # If caller sent difficulty but not task_id, derive task_id
    if task_id is None and difficulty is not None:
        task_id = _DIFF_TO_TASK.get(difficulty.lower(), "task_easy")

    # Final fallback
    if task_id is None:
        task_id = "task_easy"

    return env.reset(task_id=task_id, seed=seed)


@app.post("/step", tags=["OpenEnv"], response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    """Submit one action. Returns observation, reward, done, info."""
    result = env.step(req.action)
    return result


@app.get("/state", tags=["OpenEnv"])
def state() -> Dict[str, Any]:
    """Return the full internal CrisisState as a flat dict."""
    return env.get_full_state()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (mandatory for platform validator)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()