"""
policy_model.py — Hierarchical masked policy network for the Crisis Response environment.

Hierarchy:
1) Strategy head: rescue_first | predict_first | balanced
2) Action head: classify | predict | allocate | coordinate | rescue | skip | delay
3) Parameter heads: threat/resource/zone/units/severity/tti/pop

Action masking is applied at sampling time to avoid invalid or meaningless actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from utils import (
    ACTION_TYPES,
    MAX_RESOURCES,
    MAX_RESCUE_UNITS,
    MAX_THREATS,
    MAX_ZONES,
    STATE_DIM,
    STRATEGY_TYPES,
    build_state_vector,
    build_valid_action_mask,
    decode_action,
    observation_to_dict,
)


@dataclass(frozen=True)
class PolicyShape:
    state_dim: int
    hidden_dim: int


STRATEGY_BIAS: Dict[str, Dict[str, float]] = {
    "rescue_first": {
        "rescue":     0.40,
        "allocate":   0.25,
        "predict":   0.15,
        "classify":  0.10,
        "coordinate": 0.05,
    },
    "predict_first": {
        "predict":    0.45,
        "classify":   0.30,
        "coordinate": 0.20,
        "allocate":   0.10,
        "rescue":    0.05,
    },
    "balanced": {
        "classify":   0.18,
        "predict":   0.18,
        "allocate":  0.18,
        "coordinate": 0.18,
        "rescue":    0.18,
    },
}


class PolicyNetwork(nn.Module):
    """Hierarchical actor with explicit action masking."""

    NUM_STRATEGIES = len(STRATEGY_TYPES)
    NUM_ACTION_TYPES = len(ACTION_TYPES)

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 512,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.shape = PolicyShape(state_dim=state_dim, hidden_dim=hidden_dim)

        self.input_proj = nn.Linear(state_dim, hidden_dim)

        self.enc_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.enc_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.value_head = nn.Linear(hidden_dim, 1)

        self.strategy_head    = nn.Linear(hidden_dim, self.NUM_STRATEGIES)
        self.action_type_head = nn.Linear(hidden_dim, self.NUM_ACTION_TYPES)
        self.threat_head      = nn.Linear(hidden_dim, MAX_THREATS)
        self.resource_head    = nn.Linear(hidden_dim, MAX_RESOURCES)
        self.zone_head        = nn.Linear(hidden_dim, MAX_ZONES)
        self.units_head       = nn.Linear(hidden_dim, MAX_RESCUE_UNITS)
        self.severity_head    = nn.Linear(hidden_dim, 10)
        self.tti_head         = nn.Linear(hidden_dim, 20)
        self.pop_head         = nn.Linear(hidden_dim, 20)

    def forward(self, state_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        h = self.input_proj(state_tensor)
        h = self.enc_block1(h) + self.skip_proj(h)
        h = self.enc_block2(h) + h

        return {
            "strategy":    self.strategy_head(h),
            "action_type": self.action_type_head(h),
            "threat":      self.threat_head(h),
            "resource":    self.resource_head(h),
            "zone":        self.zone_head(h),
            "units":       self.units_head(h),
            "severity":    self.severity_head(h),
            "tti":         self.tti_head(h),
            "pop":         self.pop_head(h),
            "value":       self.value_head(h),
        }

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool, device=logits.device)
        if mask.numel() != logits.numel():
            raise ValueError(f"Mask shape mismatch: logits={tuple(logits.shape)} mask={tuple(mask.shape)}")
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)
        masked = logits.clone()
        masked[~mask] = -1e9
        return masked

    def _sample_from_logits(self, logits: torch.Tensor, greedy: bool) -> Tuple[int, torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=logits)
        if greedy:
            idx = int(torch.argmax(logits).item())
            log_prob = torch.tensor(0.0, device=logits.device)
        else:
            sample = dist.sample()
            idx = int(sample.item())
            log_prob = dist.log_prob(sample)
        entropy = dist.entropy()
        return idx, log_prob, entropy

    @staticmethod
    def _obs_priority(threat: Dict[str, Any], strategy: str) -> float:
        sev = float(threat.get("severity", 0.0))
        pop = float(threat.get("population_at_risk", 0.0))
        tti = max(float(threat.get("time_to_impact", 1.0)), 1.0)
        unc = float(threat.get("severity_uncertainty", 0.0)) + float(threat.get("population_uncertainty", 0.0))
        return (sev * pop) / tti * (1.0 + unc * 0.10)

    @staticmethod
    def _best_resource_for_threat(threat: Dict[str, Any], resources: list[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not resources:
            return None
        zone = threat.get("zone")
        affinity = {
            "military": {"military_unit", "medical_team"},
            "maritime": {"coast_guard", "rescue_drone"},
            "urban": {"swat_team", "fire_brigade", "evacuation_bus"},
            "rural": {"fire_brigade", "medical_team", "rescue_drone"},
        }.get(zone, set())

        def score(r: Dict[str, Any]) -> float:
            base = float(r.get("effectiveness", 0.0))
            zone_bonus = 0.5 if r.get("resource_type") in affinity else 0.0
            return base + zone_bonus  # BOOSTED +0.5 zone match

        return max(resources, key=score)

    def _heuristic_greedy_action(self, obs_dict: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        mask = build_valid_action_mask(obs_dict)

        threats = [t for t in obs_dict.get("threats", []) if t.get("status") == "active"]
        resources = [r for r in obs_dict.get("resources", []) if r.get("is_available", False)]
        zones = [z for z in obs_dict.get("affected_zones", []) if z.get("is_active", False)]
        budget = int(obs_dict.get("resource_budget_remaining", 0))

        ranked = sorted(threats, key=lambda t: self._obs_priority(t, strategy), reverse=True)

        can = lambda name: mask["action_mask"][ACTION_TYPES.index(name)] == 1

        # 1) Classify unclassified active threats.
        if can("classify"):
            for threat in ranked:
                if threat.get("predicted_severity") is None:
                    return {
                        "action_type": "classify",
                        "strategy": strategy,
                        "classification": {
                            "threat_id": int(threat["threat_id"]),
                            "predicted_type": threat.get("threat_type", "airstrike"),
                            "predicted_severity": float(threat.get("severity", 5.0)),
                        },
                    }

        # 2) Predict threats missing forecast.
        if can("predict"):
            for threat in ranked:
                if threat.get("predicted_tti") is None or threat.get("predicted_pop") is None:
                    # Defensive: ensure numeric types
                    tid = threat.get("threat_id")
                    tti = threat.get("time_to_impact", 0)
                    pop = threat.get("population_at_risk", 0)
                    try:
                        tid_val = int(tid) if isinstance(tid, (int, float)) else (int(tid) if isinstance(tid, str) else 1)
                    except (ValueError, TypeError):
                        tid_val = 1
                    try:
                        tti_val = int(max(0, tti)) if isinstance(tti, (int, float)) else (int(tti) if isinstance(tti, str) else 0)
                    except (ValueError, TypeError):
                        tti_val = 0
                    try:
                        pop_val = int(max(0, pop)) if isinstance(pop, (int, float)) else (int(pop) if isinstance(pop, str) else 0)
                    except (ValueError, TypeError):
                        pop_val = 0
                    return {
                        "action_type": "predict",
                        "strategy": strategy,
                        "prediction": {
                            "threat_id": tid_val,
                            "predicted_tti": tti_val,
                            "predicted_pop": pop_val,
                        },
                    }

        # 3) Rescue currently impacted zones.
        if can("rescue") and zones and budget > 0:
            zone = max(zones, key=lambda z: int(z.get("total_victims", 0)) - int(z.get("rescued", 0)))
            remaining = int(zone.get("total_victims", 0)) - int(zone.get("rescued", 0))
            if remaining > 140:
                units = 5
            elif remaining > 80:
                units = 4
            elif remaining > 30:
                units = 3
            else:
                units = 2
            units = max(1, min(units, MAX_RESCUE_UNITS, budget))
            return {
                "action_type": "rescue",
                "strategy": strategy,
                "rescue": {"zone_id": int(zone["zone_id"]), "rescue_units_to_send": units},
            }

        # 4) Allocate before impact for imminent unassigned active threats.
        if can("allocate") and ranked and resources and budget > 0:
            # Defensive: handle potential non-numeric time_to_impact
            imminent = []
            for t in ranked:
                try:
                    tti = t.get("time_to_impact", 99)
                    if isinstance(tti, (int, float)):
                        if int(tti) <= 5 and t.get("assigned_resource") is None:
                            imminent.append(t)
                except (ValueError, TypeError):
                    pass
            if imminent:
                target = imminent[0]
                resource = self._best_resource_for_threat(target, resources)
                if resource is not None:
                    return {
                        "action_type": "allocate",
                        "strategy": strategy,
                        "allocation": {
                            "threat_id": int(target["threat_id"]),
                            "resource_id": int(resource["resource_id"]),
                        },
                    }

        # 5) Coordinate once predictions exist to lock global priority.
        if can("coordinate") and len(ranked) >= 2:
            needs_coord = any(t.get("priority_rank") is None for t in ranked)
            if needs_coord:
                return {
                    "action_type": "coordinate",
                    "strategy": strategy,
                    "coordination": {"priority_order": [int(t["threat_id"]) for t in ranked]},
                }

        # 6) Allocate best matching resource to highest unassigned risk.
        if can("allocate") and ranked and resources and budget > 0:
            target = next((t for t in ranked if t.get("assigned_resource") is None), ranked[0])
            resource = self._best_resource_for_threat(target, resources)
            if resource is not None:
                return {
                    "action_type": "allocate",
                    "strategy": strategy,
                    "allocation": {
                        "threat_id": int(target["threat_id"]),
                        "resource_id": int(resource["resource_id"]),
                    },
                }

        # 7) Coordinate ranked priorities when no other urgent action exists.
        if can("coordinate") and len(ranked) >= 2:
            return {
                "action_type": "coordinate",
                "strategy": strategy,
                "coordination": {"priority_order": [int(t["threat_id"]) for t in ranked]},
            }

        # 8) Delay near-term critical threat if still active.
        if can("delay"):
            # Defensive: handle potential non-numeric time_to_impact
            imminent = []
            for t in ranked:
                try:
                    tti = t.get("time_to_impact", 99)
                    if isinstance(tti, (int, float)) and int(tti) <= 2:
                        imminent.append(t)
                except (ValueError, TypeError):
                    pass
            if imminent:
                return {
                    "action_type": "delay",
                    "strategy": strategy,
                    "delay": {"threat_id": int(imminent[0]["threat_id"]), "delay_steps": 1},
                }

        return {"action_type": "skip", "strategy": strategy}

    def select_action(
        self,
        observation: Any,
        greedy: bool = False,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        observation -> action_dict, log_prob, entropy, logits_dict.
        """
        obs_dict = observation_to_dict(observation)
        mask = build_valid_action_mask(obs_dict)

        device = next(self.parameters()).device
        state_vec = build_state_vector(obs_dict)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device)

        with torch.set_grad_enabled(not greedy):
            logits = self.forward(state_tensor)

        # Remove batch dim.
        flat_logits = {k: v.squeeze(0) for k, v in logits.items()}

        if greedy:
            strategy_mask_t = torch.tensor(mask["strategy_mask"], dtype=torch.float32, device=device)
            strategy_logits = self._masked_logits(flat_logits["strategy"], strategy_mask_t)
            strategy_idx = int(torch.argmax(strategy_logits).item())
            strategy = STRATEGY_TYPES[strategy_idx]
            action_dict = self._heuristic_greedy_action(obs_dict, strategy)
            zero = torch.tensor(0.0, device=device)
            return action_dict, zero, zero, logits

        total_log_prob = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)

        # 1) Strategy
        strategy_mask_t = torch.tensor(mask["strategy_mask"], dtype=torch.float32, device=device)
        strategy_logits = self._masked_logits(flat_logits["strategy"], strategy_mask_t)
        strategy_idx, lp, ent = self._sample_from_logits(strategy_logits, greedy)
        total_log_prob = total_log_prob + lp
        total_entropy = total_entropy + ent
        strategy = STRATEGY_TYPES[strategy_idx]

        # 2) Action type with strategy bias + action mask
        action_logits = flat_logits["action_type"].clone()
        bias_cfg = STRATEGY_BIAS.get(strategy, {})
        for action_name, bonus in bias_cfg.items():
            if action_name in ACTION_TYPES:
                action_logits[ACTION_TYPES.index(action_name)] += float(bonus)

        action_mask_t = torch.tensor(mask["action_mask"], dtype=torch.float32, device=device)
        action_logits = self._masked_logits(action_logits, action_mask_t)

        action_idx, lp, ent = self._sample_from_logits(action_logits, greedy)
        total_log_prob = total_log_prob + lp
        total_entropy = total_entropy + ent

        action_name = ACTION_TYPES[action_idx]

        # 3) Sample relevant low-level arguments only
        threat_idx = 0
        resource_idx = 0
        zone_idx = 0
        units_idx = 0
        severity_idx = 5
        tti_idx = 10
        pop_idx = 10

        if action_name in {"classify", "predict", "allocate", "delay"}:
            threat_mask_t = torch.tensor(mask["threat_mask"], dtype=torch.float32, device=device)
            threat_logits = self._masked_logits(flat_logits["threat"], threat_mask_t)
            threat_idx, lp, ent = self._sample_from_logits(threat_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent

        if action_name == "allocate":
            resource_mask_t = torch.tensor(mask["resource_mask"], dtype=torch.float32, device=device)
            resource_logits = self._masked_logits(flat_logits["resource"], resource_mask_t)
            resource_idx, lp, ent = self._sample_from_logits(resource_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent

        if action_name == "rescue":
            zone_mask_t = torch.tensor(mask["zone_mask"], dtype=torch.float32, device=device)
            zone_logits = self._masked_logits(flat_logits["zone"], zone_mask_t)
            zone_idx, lp, ent = self._sample_from_logits(zone_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent

            units_mask_t = torch.tensor(mask["units_mask"], dtype=torch.float32, device=device)
            units_logits = self._masked_logits(flat_logits["units"], units_mask_t)
            units_idx, lp, ent = self._sample_from_logits(units_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent

        if action_name == "delay":
            units_mask_t = torch.tensor([1.0] * MAX_RESCUE_UNITS, dtype=torch.float32, device=device)
            units_logits = self._masked_logits(flat_logits["units"], units_mask_t)
            units_idx, lp, ent = self._sample_from_logits(units_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent

        if action_name == "classify":
            severity_logits = flat_logits["severity"]
            severity_idx, lp, ent = self._sample_from_logits(severity_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent

        if action_name == "predict":
            tti_logits = flat_logits["tti"]
            pop_logits = flat_logits["pop"]
            tti_idx, lp, ent = self._sample_from_logits(tti_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent
            pop_idx, lp, ent = self._sample_from_logits(pop_logits, greedy)
            total_log_prob = total_log_prob + lp
            total_entropy = total_entropy + ent

        action_dict = decode_action(
            obs_dict=obs_dict,
            strategy_idx=strategy_idx,
            action_type_idx=action_idx,
            threat_idx=threat_idx,
            resource_idx=resource_idx,
            zone_idx=zone_idx,
            units_idx=units_idx,
            severity_idx=severity_idx,
            tti_idx=tti_idx,
            pop_idx=pop_idx,
        )

        return action_dict, total_log_prob, total_entropy, logits
