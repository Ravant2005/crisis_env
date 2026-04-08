"""
grader.py — Task graders for CrisisAI OpenEnv environment.

The evaluator calls these functions to score each task.
Each function returns a float in [0.0, 1.0].
"""

import requests
from typing import Any, Dict, Optional

ENV_URL = "https://praveen4278-crisis-ai-env.hf.space"


def _get_scores(session_id: str = "default") -> Dict[str, float]:
    """Fetch current task scores from the environment server."""
    try:
        r = requests.get(f"{ENV_URL}/scores", params={"session_id": session_id}, timeout=10)
        data = r.json()
        if "status" in data and data["status"] == "error":
            return {}
        return data
    except Exception:
        return {}


def grade_task_easy(env: Any = None, session_id: str = "default", **kwargs) -> float:
    """Grade Task 1: Threat Classification (easy difficulty). Returns score in [0, 1]."""
    scores = _get_scores(session_id)
    return float(scores.get("classification", 0.0))


def grade_task_medium(env: Any = None, session_id: str = "default", **kwargs) -> float:
    """Grade Task 2: Impact Prediction (medium difficulty). Returns score in [0, 1]."""
    scores = _get_scores(session_id)
    return float(scores.get("prediction", 0.0))


def grade_task_medium_plus(env: Any = None, session_id: str = "default", **kwargs) -> float:
    """Grade Task 3: Resource Allocation (medium+ difficulty). Returns score in [0, 1]."""
    scores = _get_scores(session_id)
    return float(scores.get("allocation", 0.0))


def grade_task_hard(env: Any = None, session_id: str = "default", **kwargs) -> float:
    """Grade Task 4: Multi-Threat Coordination (hard difficulty). Returns score in [0, 1]."""
    scores = _get_scores(session_id)
    return float(scores.get("coordination", 0.0))


def grade_task_advanced(env: Any = None, session_id: str = "default", **kwargs) -> float:
    """Grade Task 5: Rescue Optimisation (advanced difficulty). Returns score in [0, 1]."""
    scores = _get_scores(session_id)
    return float(scores.get("rescue", 0.0))


def grade_final(env: Any = None, session_id: str = "default", **kwargs) -> float:
    """Compute final weighted score across all tasks. Returns score in [0, 1]."""
    scores = _get_scores(session_id)
    return float(scores.get("final_score", scores.get("final", 0.0)))


# Aliases the evaluator may use
grade_classification = grade_task_easy
grade_prediction     = grade_task_medium
grade_allocation     = grade_task_medium_plus
grade_coordination   = grade_task_hard
grade_rescue         = grade_task_advanced
