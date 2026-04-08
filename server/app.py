from __future__ import annotations
from typing import Dict, Any, Optional
from fastapi import FastAPI
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
    docs_url="/",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

env = CrisisEnvEnvironment()

# ──────────────────────────────────────────────────────────────────────────────
# 🧠 Pydantic Models (CRITICAL)
# ──────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"
    difficulty: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class Observation(BaseModel):
    state: Dict[str, Any]


class StepResult(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────────────
# 🔹 SYSTEM endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", tags=["System"])
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": info["id"],
                "name": info["name"],
                "difficulty": info["difficulty"],
                "description": info["description"],
                "grader_range": info["grader_range"],
                "inbox_size": info.get("inbox_size", 10),
            }
            for info in TASK_REGISTRY.values()
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# 🔹 OPENENV endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/reset", tags=["OpenEnv"], response_model=Observation)
def reset(req: ResetRequest):
    result = env.reset(
        task_id=req.task_id,
        seed=req.seed
    )
    return {"state": result}


@app.post("/step", tags=["OpenEnv"], response_model=StepResult)
def step(req: StepRequest):
    result = env.step(req.action)
    return result


@app.get("/state", tags=["OpenEnv"], response_model=Observation)
def state():
    result = env.get_full_state()
    return {"state": result}


# ──────────────────────────────────────────────────────────────────────────────
# 🚀 ENTRY POINT (MANDATORY for validator)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()