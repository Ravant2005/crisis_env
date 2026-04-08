from uuid import uuid4
from typing import Any, Dict, Optional

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    Environment = object
    State = None

from server.environment import CrisisEnvironment
from models import CrisisAction


class CrisisEnvEnvironment(Environment):
    """
    OpenEnv Wrapper for CrisisEnvironment.
    Ensures compatibility with the OpenEnv interface by handling JSON serialization
    and mapping internal methods to the expected API.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self.env = CrisisEnvironment()
        self._state = None
        if State:
            self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, task_id: str = "task_easy", seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment and return the initial observation as a dict."""
        obs = self.env.reset(seed=seed)

        if State:
            self._state = State(
                episode_id=str(uuid4()),
                step_count=0,
            )
        
        # OpenEnv expects a plain dict for JSON serialization
        return obs.model_dump()

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step in the environment using a raw action dictionary."""
        # Fix Action Format Mismatch: Parse raw dict into CrisisAction model
        action = CrisisAction(**action_dict)
        result = self.env.step(action)

        if self._state:
            self._state.step_count += 1

        # Fix JSON Serialization Bug: Always return .model_dump() (Pydantic v2)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }

    @property
    def state(self):
        """Return the current OpenEnv state."""
        return self._state

    def task_scores(self) -> Dict[str, float]:
        """Return the internal task scores from the simulation logic."""
        return self.env.task_scores()

    def get_full_state(self) -> Dict[str, Any]:
        """Return the full internal state as a dict."""
        return self.env.state().model_dump()
