"""
server/app.py — CrisisAI OpenEnv Environment Server.

Uses openenv.core.env_server.http_server.create_app() to create a
fully compliant OpenEnv server with all required endpoints including
/health, /metadata, /schema, /mcp, /tasks with graders.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict, List, Optional
from pydantic import Field

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.interfaces import Environment
from openenv.core.rubrics import Rubric

# ─────────────────────────────────────────────
# IMPORT CRISIS ENVIRONMENT
# ─────────────────────────────────────────────

try:
    from server.environment import CrisisEnvironment
    from models import CrisisAction as _CrisisAction
except ImportError:
    from environment import CrisisEnvironment
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import CrisisAction as _CrisisAction


# ─────────────────────────────────────────────
# OPENENV ACTION / OBSERVATION MODELS
# ─────────────────────────────────────────────

class CrisisEnvAction(Action):
    """Action for CrisisAI environment."""
    action_type: str = Field(default="skip", description="Action type")
    classification: Optional[Dict[str, Any]] = Field(default=None)
    prediction:     Optional[Dict[str, Any]] = Field(default=None)
    allocation:     Optional[Dict[str, Any]] = Field(default=None)
    coordination:   Optional[Dict[str, Any]] = Field(default=None)
    rescue:         Optional[Dict[str, Any]] = Field(default=None)
    delay:          Optional[Dict[str, Any]] = Field(default=None)


class CrisisEnvObservation(Observation):
    """Observation from CrisisAI environment."""
    threats:                   List[Dict[str, Any]] = Field(default_factory=list)
    resources:                 List[Dict[str, Any]] = Field(default_factory=list)
    affected_zones:            List[Dict[str, Any]] = Field(default_factory=list)
    time_remaining:            int                  = Field(default=0)
    resource_budget_remaining: int                  = Field(default=0)
    resource_budget_total:     int                  = Field(default=0)
    alerts:                    List[str]            = Field(default_factory=list)
    episode_id:                str                  = Field(default="")
    valid_actions:             Dict[str, Any]       = Field(default_factory=dict)
    # Task scores embedded in observation
    classification_score: float = Field(default=0.0)
    prediction_score:     float = Field(default=0.0)
    allocation_score:     float = Field(default=0.0)
    coordination_score:   float = Field(default=0.0)
    rescue_score:         float = Field(default=0.0)
    final_score:          float = Field(default=0.0)


# ─────────────────────────────────────────────
# GRADERS (Rubrics)
# ─────────────────────────────────────────────

class ClassificationRubric(Rubric):
    """Grader for Threat Classification — score in [0, 1]."""
    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(observation, CrisisEnvObservation):
            return float(observation.classification_score)
        if isinstance(observation, dict):
            return float(observation.get("classification_score", 0.0))
        return 0.0


class PredictionRubric(Rubric):
    """Grader for Impact Prediction — score in [0, 1]."""
    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(observation, CrisisEnvObservation):
            return float(observation.prediction_score)
        if isinstance(observation, dict):
            return float(observation.get("prediction_score", 0.0))
        return 0.0


class AllocationRubric(Rubric):
    """Grader for Resource Allocation — score in [0, 1]."""
    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(observation, CrisisEnvObservation):
            return float(observation.allocation_score)
        if isinstance(observation, dict):
            return float(observation.get("allocation_score", 0.0))
        return 0.0


class CoordinationRubric(Rubric):
    """Grader for Multi-Threat Coordination — score in [0, 1]."""
    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(observation, CrisisEnvObservation):
            return float(observation.coordination_score)
        if isinstance(observation, dict):
            return float(observation.get("coordination_score", 0.0))
        return 0.0


class RescueRubric(Rubric):
    """Grader for Rescue Optimisation — score in [0, 1]."""
    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(observation, CrisisEnvObservation):
            return float(observation.rescue_score)
        if isinstance(observation, dict):
            return float(observation.get("rescue_score", 0.0))
        return 0.0


class CrisisRubric(Rubric):
    """Composite rubric with 5 named graders for all crisis tasks."""
    def __init__(self):
        super().__init__()
        self.classification = ClassificationRubric()
        self.prediction      = PredictionRubric()
        self.allocation      = AllocationRubric()
        self.coordination    = CoordinationRubric()
        self.rescue          = RescueRubric()

    def forward(self, action: Any, observation: Any) -> float:
        c  = self.classification(action, observation)
        p  = self.prediction(action, observation)
        a  = self.allocation(action, observation)
        co = self.coordination(action, observation)
        r  = self.rescue(action, observation)
        return round(0.20*c + 0.20*p + 0.20*a + 0.15*co + 0.25*r, 4)


# ─────────────────────────────────────────────
# OPENENV ENVIRONMENT WRAPPER
# ─────────────────────────────────────────────

class CrisisEnvWrapper(Environment):
    """
    OpenEnv-compliant wrapper around CrisisEnvironment.
    Exposes 5 named rubric graders for task validation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__(rubric=CrisisRubric())
        self._env  = CrisisEnvironment(seed=42)
        self._state = State(episode_id="", step_count=0)

    def reset(self, seed: Optional[int] = None, **kwargs) -> CrisisEnvObservation:
        obs = self._env.reset(seed=seed or 42, difficulty="medium")
        self._state = State(episode_id=obs.episode_id, step_count=0)
        return self._obs_to_openenv(obs)

    def step(self, action: CrisisEnvAction) -> CrisisEnvObservation:  # type: ignore[override]
        crisis_action = _CrisisAction(**action.model_dump(exclude={"metadata"}))
        result = self._env.step(crisis_action)
        self._state.step_count += 1
        obs = self._obs_to_openenv(result.observation)
        obs.reward = result.reward
        obs.done   = result.done
        # Embed task scores into observation
        scores = self._env.task_scores()
        obs.classification_score = scores.get("classification", 0.0)
        obs.prediction_score     = scores.get("prediction",     0.0)
        obs.allocation_score     = scores.get("allocation",     0.0)
        obs.coordination_score   = scores.get("coordination",   0.0)
        obs.rescue_score         = scores.get("rescue",         0.0)
        obs.final_score          = self._env.state().final_score
        return obs

    @property
    def state(self) -> State:
        return self._state

    def _obs_to_openenv(self, obs) -> CrisisEnvObservation:
        d = obs.model_dump()
        return CrisisEnvObservation(
            threats                   = d.get("threats", []),
            resources                 = d.get("resources", []),
            affected_zones            = d.get("affected_zones", []),
            time_remaining            = d.get("time_remaining", 0),
            resource_budget_remaining = d.get("resource_budget_remaining", 0),
            resource_budget_total     = d.get("resource_budget_total", 0),
            alerts                    = d.get("alerts", []),
            episode_id                = d.get("episode_id", ""),
            valid_actions             = d.get("valid_actions", {}),
        )


# ─────────────────────────────────────────────
# CREATE APP via openenv create_app()
# ─────────────────────────────────────────────

app = create_app(
    CrisisEnvWrapper,
    CrisisEnvAction,
    CrisisEnvObservation,
    env_name="crisis_env",
    max_concurrent_envs=4,
)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
