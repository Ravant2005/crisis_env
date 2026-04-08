from __future__ import annotations
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body
from server.crisis_env_environment import CrisisEnvEnvironment

app = FastAPI(title="CrisisAI OpenEnv Environment")

# Initialize the mandatory wrapper
env = CrisisEnvEnvironment()

@app.get("/health")
def health() -> Dict[str, str]:
    """Basic health check for OpenEnv compliance."""
    return {"status": "ok"}

@app.post("/reset")
def reset(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Reset environment and return initial observation."""
    task_id = body.get("task_id", "task_easy")
    seed = body.get("seed", None)
    return env.reset(task_id=task_id, seed=seed)

@app.post("/step")
def step(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Execute one step in the environment."""
    # OpenEnv sends { "action": { ... } }
    action_dict = body.get("action", {})
    return env.step(action_dict)

@app.get("/state")
def state() -> Dict[str, Any]:
    """Expose the full internal environment state."""
    return env.get_full_state()

@app.get("/scores")
def scores() -> Dict[str, float]:
    """Expose the current task scores directly for grading."""
    return env.task_scores()

@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    """List available tasks for this environment."""
    return {
        "tasks": [
            {"id": "task_easy", "description": "Classification focused", "grader_range": [0.0, 1.0]},
            {"id": "task_medium", "description": "Prediction focused", "grader_range": [0.0, 1.0]},
            {"id": "task_medium_plus", "description": "Allocation focused", "grader_range": [0.0, 1.0]},
            {"id": "task_hard", "description": "Coordination focused", "grader_range": [0.0, 1.0]},
            {"id": "task_advanced", "description": "Rescue focused", "grader_range": [0.0, 1.0]},
        ]
    }
