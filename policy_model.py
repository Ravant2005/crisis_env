"""
policy_model.py — Multi-head policy network for the Crisis Response Environment.

Architecture:
  State → Shared Encoder → Action Type Head  (5 logits)
                           → Threat Head       (3 logits)
                           → Resource Head     (8 logits)
                           → Zone Head         (3 logits)
                           → Units Head        (5 logits)
                           → Severity Head     (10 logits)
                           → TTI Head          (20 logits)
                           → Population Head   (20 logits)
                           → Threat Type Head  (6 logits)

The agent DIRECTLY outputs all action components — no pre-computed
candidates or heuristic optimal solutions are injected.

Coordinate priority ordering is derived from threat_logits (sorted by
logit value), giving the model a learnable ordering mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PolicyShape:
    state_dim: int
    hidden_dim: int


class PolicyNetwork(nn.Module):
    """
    Multi-head policy that directly produces action components from state.

    During training: samples from Categorical(logits) for each component.
    During evaluation: uses argmax for deterministic selection.

    Coordinate actions use threat_logits sorted by value as priority ordering.
    """

    NUM_ACTION_TYPES = 5   # classify, predict, allocate, coordinate, rescue
    NUM_THREATS      = 3
    NUM_RESOURCES    = 8
    NUM_ZONES        = 3
    NUM_UNITS        = 5   # 1..5 rescue units
    NUM_SEVERITY     = 10  # bins for severity prediction
    NUM_TTI          = 20  # bins for time-to-impact prediction
    NUM_POP          = 20  # bins for population prediction
    NUM_THREAT_TYPES = 6   # airstrike, ship_attack, etc.

    def __init__(
        self,
        state_dim: int = 229,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shape = PolicyShape(state_dim=state_dim, hidden_dim=hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Action component heads
        self.action_type_head  = nn.Linear(hidden_dim, self.NUM_ACTION_TYPES)
        self.threat_head       = nn.Linear(hidden_dim, self.NUM_THREATS)
        self.resource_head     = nn.Linear(hidden_dim, self.NUM_RESOURCES)
        self.zone_head         = nn.Linear(hidden_dim, self.NUM_ZONES)
        self.units_head        = nn.Linear(hidden_dim, self.NUM_UNITS)
        self.severity_head     = nn.Linear(hidden_dim, self.NUM_SEVERITY)
        self.tti_head          = nn.Linear(hidden_dim, self.NUM_TTI)
        self.pop_head          = nn.Linear(hidden_dim, self.NUM_POP)
        self.threat_type_head  = nn.Linear(hidden_dim, self.NUM_THREAT_TYPES)

    def forward(self, state_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_tensor: [state_dim] or [batch, state_dim]
        Returns:
            dict of logits for each action component
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        h = self.encoder(state_tensor)

        return {
            "action_type":  self.action_type_head(h),
            "threat":       self.threat_head(h),
            "resource":     self.resource_head(h),
            "zone":         self.zone_head(h),
            "units":        self.units_head(h),
            "severity":     self.severity_head(h),
            "tti":          self.tti_head(h),
            "pop":          self.pop_head(h),
            "type":         self.threat_type_head(h),
        }

    def select_action(
        self,
        observation: Any,
        greedy: bool = False,
    ) -> Tuple[Dict[str, Any], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        End-to-end: observation → (action_dict, log_prob, logits).

        Args:
            observation: CrisisObservation or dict
            greedy: if True, use argmax instead of sampling (evaluation)

        Returns:
            action_dict: valid CrisisAction-compatible dict
            log_prob: scalar total log probability (for REINFORCE)
            logits: all raw logits (for entropy computation)
        """
        from utils import build_state_vector, decode_action, observation_to_dict

        obs_dict = observation_to_dict(observation)
        state_vec = build_state_vector(obs_dict)
        device = next(self.parameters()).device
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device)

        with torch.set_grad_enabled(not greedy):
            logits = self.forward(state_tensor)

        if greedy:
            samples = {k: v.argmax(dim=-1).item() for k, v in logits.items()}
        else:
            samples = {}
            for k, v in logits.items():
                dist = Categorical(logits=v.squeeze(0))
                samples[k] = dist.sample().item()

        # Build coordinate priority from threat logits (learned ordering)
        threat_logits_squeezed = logits["threat"].squeeze(0)
        threat_order = torch.argsort(threat_logits_squeezed, descending=True).tolist()

        # Decode action
        action_dict = decode_action(
            obs_dict=obs_dict,
            action_type_idx=samples["action_type"],
            threat_idx=samples["threat"],
            resource_idx=samples["resource"],
            zone_idx=samples["zone"],
            units_idx=samples["units"],
            severity_idx=samples["severity"],
            tti_idx=samples["tti"],
            pop_idx=samples["pop"],
        )

        # Override coordinate priority with learned ordering
        if samples["action_type"] == 3:  # coordinate
            active_threats = sorted(
                [t for t in obs_dict.get("threats", []) if t.get("status") == "active"],
                key=lambda t: t["threat_id"],
            )
            active_ids = [t["threat_id"] for t in active_threats]
            # Map threat logits order to actual threat IDs
            priority = [active_ids[i] for i in threat_order if i < len(active_ids)]
            action_dict = {
                "action_type": "coordinate",
                "coordination": {"priority_order": priority},
            }

        # Compute log probability
        log_prob = self._compute_log_prob(logits, samples)

        return action_dict, log_prob, logits

    def _compute_log_prob(
        self,
        logits: Dict[str, torch.Tensor],
        samples: Dict[str, int],
    ) -> torch.Tensor:
        """Sum of log probs across all sampled action components."""
        device = next(iter(logits.values())).device
        log_prob = torch.tensor(0.0, device=device)
        for key in ["action_type", "threat", "resource", "zone", "units",
                     "severity", "tti", "pop", "type"]:
            dist = Categorical(logits=logits[key].squeeze(0))
            log_prob = log_prob + dist.log_prob(torch.tensor(samples[key], device=device))
        return log_prob

    def entropy(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Total entropy across all action component distributions."""
        device = next(iter(logits.values())).device
        total = torch.tensor(0.0, device=device)
        for v in logits.values():
            dist = Categorical(logits=v.squeeze(0))
            total = total + dist.entropy()
        return total
