"""
client.py — Python client SDK for the AI Crisis Response & Rescue Coordination Environment.
Provides a clean, synchronous interface for interacting with the running server.

Usage:
    from client import CrisisEnvClient

    client = CrisisEnvClient("http://localhost:8000")
    obs    = client.reset(seed=42)
    result = client.step(action_dict)
    state  = client.state()
    scores = client.scores()
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class CrisisEnvClient:
    """
    Synchronous HTTP client for the Crisis Response OpenEnv server.
    All methods return plain dicts matching the server's JSON responses.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout:  int = 30,
        hf_token: Optional[str] = None,
    ):
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout
        self._session   = requests.Session()
        if hf_token:
            self._session.headers.update({"Authorization": f"Bearer {hf_token}"})
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Core OpenEnv API ──────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, session_id: str = "default") -> Dict[str, Any]:
        """Reset the environment. Returns the initial observation dict."""
        payload: Dict[str, Any] = {"session_id": session_id}
        if seed is not None:
            payload["seed"] = seed
        return self._post("/reset", payload)

    def step(self, action: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
        """
        Submit one action. Returns {observation, reward, done, info}.

        action must match CrisisAction schema, e.g.:
            {
                "action_type": "classify",
                "classification": {
                    "threat_id": 1,
                    "predicted_type": "airstrike",
                    "predicted_severity": 7.5
                }
            }
        """
        return self._post("/step", {"action": action, "session_id": session_id})

    def state(self, session_id: str = "default") -> Dict[str, Any]:
        """Return the current episode state and all 5 grader scores."""
        return self._get("/state", params={"session_id": session_id})

    def scores(self, session_id: str = "default") -> Dict[str, Any]:
        """Return only the grader score summary (fast convenience call)."""
        return self._get("/scores", params={"session_id": session_id})

    def tasks(self) -> Dict[str, Any]:
        """Return the list of all task definitions."""
        return self._get("/tasks")

    def health(self) -> Dict[str, Any]:
        """Liveness check — returns 200 if server is up."""
        return self._get("/health")

    # ── Convenience helpers ───────────────────────────────────────────────

    def classify(self, threat_id: int, predicted_type: str, predicted_severity: float) -> Dict:
        return self.step({
            "action_type":    "classify",
            "classification": {
                "threat_id":          threat_id,
                "predicted_type":     predicted_type,
                "predicted_severity": predicted_severity,
            },
        })

    def predict(self, threat_id: int, predicted_tti: int, predicted_pop: int) -> Dict:
        return self.step({
            "action_type": "predict",
            "prediction":  {
                "threat_id":     threat_id,
                "predicted_tti": predicted_tti,
                "predicted_pop": predicted_pop,
            },
        })

    def allocate(self, threat_id: int, resource_id: int) -> Dict:
        return self.step({
            "action_type": "allocate",
            "allocation":  {
                "threat_id":   threat_id,
                "resource_id": resource_id,
            },
        })

    def coordinate(self, priority_order: list) -> Dict:
        return self.step({
            "action_type":  "coordinate",
            "coordination": {"priority_order": priority_order},
        })

    def rescue(self, zone_id: int, units: int = 5) -> Dict:
        return self.step({
            "action_type": "rescue",
            "rescue": {
                "zone_id":              zone_id,
                "rescue_units_to_send": units,
            },
        })

    # ── Internal ──────────────────────────────────────────────────────────

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._session.post(
            f"{self.base_url}{path}",
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        r = self._session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def __repr__(self) -> str:
        return f"CrisisEnvClient(base_url={self.base_url!r})"
