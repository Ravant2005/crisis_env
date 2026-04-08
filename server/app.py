from __future__ import annotations
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body
from fastapi.openapi.utils import get_openapi
from server.crisis_env_environment import CrisisEnvEnvironment
from grader import TASK_REGISTRY

# ── Tags define the SECTIONS shown in Swagger UI ──────────────────────────────
tags_metadata = [
    {
        "name": "System",
        "description": "Health checks and task listing.",
    },
    {
        "name": "OpenEnv",
        "description": "Core RL environment endpoints: reset, step, state.",
    },
]

app = FastAPI(
    title="CrisisAI OpenEnv Environment",
    version="1.0.0",
    description="OpenEnv RL environment for AI Crisis Response & Rescue Coordination. 3 difficulty tasks.",
    openapi_tags=tags_metadata,
    docs_url="/",          # ← Swagger UI appears at root URL (like your friend's)
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

env = CrisisEnvEnvironment()


# ── SYSTEM endpoints ───────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", tags=["System"])
def tasks() -> Dict[str, Any]:
    """
    Returns all registered tasks with grader metadata.
    The platform validates that at least 3 tasks have grader_range [0.0, 1.0].
    """
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


# ── OPENENV endpoints ──────────────────────────────────────────────────────────

@app.post("/reset", tags=["OpenEnv"])
def reset(body: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    body       = body or {}
    task_id    = body.get("task_id", "task_easy")
    difficulty = body.get("difficulty", TASK_REGISTRY.get(task_id, {}).get("difficulty", "medium"))
    seed       = body.get("seed", None)
    return env.reset(task_id=task_id, seed=seed)


@app.post("/step", tags=["OpenEnv"])
def step(body: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    body        = body or {}
    action_dict = body.get("action", {})
    return env.step(action_dict)


@app.get("/state", tags=["OpenEnv"])
def state() -> Dict[str, Any]:
    return env.get_full_state()


@app.get("/scores", tags=["OpenEnv"])
def scores() -> Dict[str, float]:
    raw = env.task_scores()
    final = (
        0.20 * raw.get("classification", 0.0) +
        0.20 * raw.get("prediction",     0.0) +
        0.20 * raw.get("allocation",     0.0) +
        0.15 * raw.get("coordination",   0.0) +
        0.25 * raw.get("rescue",         0.0)
    )
    return {**raw, "final": round(final, 4), "final_score": round(final, 4)}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()