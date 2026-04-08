"""
grader.py — Task graders for CrisisAI OpenEnv environment.

The evaluator calls these functions to score each task.
Each function returns a float in [0.0, 1.0].
"""

from typing import Any, Dict


def _get_scores(env: Any) -> Dict[str, float]:
    """
    Extract current task scores from the environment object.
    Supports both the wrapper and the internal simulation engine.
    """
    if env is None:
        return {}
    
    # Try the task_scores() method on the environment or its internal engine
    if hasattr(env, "task_scores"):
        return env.task_scores()
    elif hasattr(env, "env") and hasattr(env.env, "task_scores"):
        return env.env.task_scores()
    
    return {}


def grade_task_easy(env: Any = None, **kwargs) -> float:
    """Grade Task 1: Threat Classification. Returns score in [0, 1]."""
    scores = _get_scores(env)
    return float(scores.get("classification", 0.0))


def grade_task_medium(env: Any = None, **kwargs) -> float:
    """Grade Task 2: Impact Prediction. Returns score in [0, 1]."""
    scores = _get_scores(env)
    return float(scores.get("prediction", 0.0))


def grade_task_medium_plus(env: Any = None, **kwargs) -> float:
    """Grade Task 3: Resource Allocation. Returns score in [0, 1]."""
    scores = _get_scores(env)
    return float(scores.get("allocation", 0.0))


def grade_task_hard(env: Any = None, **kwargs) -> float:
    """Grade Task 4: Multi-Threat Coordination. Returns score in [0, 1]."""
    scores = _get_scores(env)
    return float(scores.get("coordination", 0.0))


def grade_task_advanced(env: Any = None, **kwargs) -> float:
    """Grade Task 5: Rescue Optimisation. Returns score in [0, 1]."""
    scores = _get_scores(env)
    return float(scores.get("rescue", 0.0))


def grade_final(env: Any = None, **kwargs) -> float:
    """Compute final weighted score across all tasks. Returns score in [0, 1]."""
    scores = _get_scores(env)
    return float(scores.get("final_score", scores.get("final", 0.0)))


# Aliases for convenience
grade_classification = grade_task_easy
grade_prediction     = grade_task_medium
grade_allocation     = grade_task_medium_plus
grade_coordination   = grade_task_hard
grade_rescue         = grade_task_advanced
