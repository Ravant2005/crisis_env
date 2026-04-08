"""
models.py — Typed Pydantic models for the AI Crisis Response & Rescue Coordination Environment.
Implements the full OpenEnv spec: Action, Observation, State, and supporting data models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────


class ActionType(str, Enum):
    CLASSIFY = "classify"
    PREDICT = "predict"
    ALLOCATE = "allocate"
    COORDINATE = "coordinate"
    RESCUE = "rescue"
    SKIP = "skip"
    DELAY = "delay"
    EVACUATE = "evacuate"  # retained for backward compatibility


class ThreatType(str, Enum):
    INDUSTRIAL_EXPLOSION = "industrial_explosion"
    COASTAL_FLOOD = "coastal_flood"
    WILDFIRE_SPREAD = "wildfire_spread"
    EXPLOSION = "explosion"
    FLOOD = "flood"
    FIRE = "fire"
    AIRSTRIKE = "airstrike"
    SHIP_ATTACK = "ship_attack"
    DRONE_THREAT = "drone_threat"


class ThreatStatus(str, Enum):
    ACTIVE = "active"
    IMPACTED = "impacted"
    CONTAINED = "contained"
    RESOLVED = "resolved"


class ResourceType(str, Enum):
    HEAVY_EQUIPMENT = "heavy_equipment"
    COAST_GUARD = "coast_guard"
    HAZMAT_TEAM = "hazmat_team"
    FIRE_BRIGADE = "fire_brigade"
    MEDICAL_TEAM = "medical_team"
    RESCUE_DRONE = "rescue_drone"
    EVACUATION_BUS = "evacuation_bus"
    MILITARY_UNIT = "military_unit"
    SWAT_TEAM = "swat_team"


class ZoneType(str, Enum):
    MILITARY = "military"
    MARITIME = "maritime"
    URBAN = "urban"
    RURAL = "rural"


# ─────────────────────────────────────────────
# SUPPORTING SUB-MODELS
# ─────────────────────────────────────────────


class ThreatInfo(BaseModel):
    """Represents a single active or resolved threat in the environment."""

    threat_id: int = Field(..., description="Unique threat identifier")
    threat_type: ThreatType = Field(..., description="Category of threat")
    status: ThreatStatus = Field(default=ThreatStatus.ACTIVE)

    # Observed values (can be noisy estimates under partial observability)
    severity: float = Field(..., ge=0.0, le=10.0, description="Observed threat severity 0–10")
    population_at_risk: int = Field(..., ge=0, description="Observed population at risk")
    time_to_impact: int = Field(..., ge=0, description="Observed steps remaining until impact")

    # Explicit uncertainty estimates exposed to the agent
    severity_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    population_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    tti_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)

    zone: ZoneType = Field(..., description="Geographic zone type")
    location_name: str = Field(..., description="Human-readable location label")

    predicted_severity: Optional[float] = Field(default=None, description="Agent's predicted severity")
    predicted_tti: Optional[int] = Field(default=None, description="Agent's predicted time-to-impact")
    predicted_pop: Optional[int] = Field(default=None, description="Agent's predicted population")
    assigned_resource: Optional[int] = Field(default=None, description="Resource ID assigned to this threat")
    priority_rank: Optional[int] = Field(default=None, description="Agent's assigned priority rank")

    casualties: int = Field(default=0, description="Casualties incurred after impact")
    casualties_prevented: int = Field(default=0, description="Casualties prevented by response")
    population_evacuated: int = Field(default=0, description="Population evacuated before impact")

    priority_score: float = Field(default=0.0, description="Estimated priority score")
    risk_level: str = Field(default="low", description="Estimated risk level")
    recommended_action_hint: str = Field(default="monitor", description="Suggested next action")


class ResourceInfo(BaseModel):
    """Represents a single deployable resource unit."""

    resource_id: int = Field(..., description="Unique resource identifier")
    resource_type: ResourceType = Field(..., description="Type of resource unit")
    is_available: bool = Field(default=True, description="Whether resource is currently free")
    assigned_to: Optional[int] = Field(default=None, description="Threat ID this resource is assigned to")
    effectiveness: float = Field(..., ge=0.0, le=1.0, description="Base effectiveness multiplier")
    location_zone: ZoneType = Field(..., description="Zone this resource is stationed in")
    cooldown_steps: int = Field(default=0, ge=0, description="Steps until resource can be reused")
    distance: float = Field(default=0.0, ge=0.0, description="Distance from incident in km")


class AffectedZoneInfo(BaseModel):
    """Post-impact zone requiring rescue operations."""

    zone_id: int = Field(..., description="Unique zone identifier")
    zone_type: ZoneType = Field(...)
    location_name: str = Field(...)
    total_victims: int = Field(default=0, description="Total victims needing rescue")
    rescued: int = Field(default=0, description="Victims already rescued")
    rescue_units_deployed: int = Field(default=0)
    is_active: bool = Field(default=False, description="True if impact occurred here")
    evacuated: int = Field(default=0, description="Population evacuated before impact")
    evacuation_units_deployed: int = Field(default=0, description="Units used for evacuation")


class ClassificationPayload(BaseModel):
    threat_id: int = Field(..., description="Which threat to classify")
    predicted_type: ThreatType = Field(..., description="Agent's predicted threat type")
    predicted_severity: float = Field(..., ge=0.0, le=10.0, description="Agent's predicted severity")


class PredictionPayload(BaseModel):
    threat_id: int = Field(..., description="Which threat to predict")
    predicted_tti: int = Field(..., ge=0, description="Predicted steps to impact")
    predicted_pop: int = Field(..., ge=0, description="Predicted population affected")


class AllocationPayload(BaseModel):
    threat_id: int = Field(..., description="Target threat")
    resource_id: int = Field(..., description="Resource to assign")


class CoordinationPayload(BaseModel):
    priority_order: List[int] = Field(..., description="Threat IDs from highest to lowest priority")


class RescuePayload(BaseModel):
    zone_id: int = Field(..., description="Affected zone to rescue")
    rescue_units_to_send: int = Field(..., ge=1, description="Number of rescue units to dispatch")


class DelayPayload(BaseModel):
    threat_id: int = Field(..., description="Target active threat")
    delay_steps: int = Field(default=1, ge=1, le=3, description="Requested delay steps")


class EvacuationPayload(BaseModel):
    zone_id: int = Field(..., description="Zone to evacuate")
    evac_units: int = Field(..., ge=1, description="Number of evacuation units")
    population_move: int = Field(default=0, description="Estimated population moved to safety")


# ─────────────────────────────────────────────
# CORE OpenEnv MODELS
# ─────────────────────────────────────────────


class CrisisAction(BaseModel):
    """OpenEnv Action model. Agent submits one action per step."""

    action_type: ActionType = Field(..., description="Which action the agent is performing")

    # Optional high-level strategy hint for hierarchical policies
    strategy: Optional[str] = Field(default=None, description="High-level strategy label")

    classification: Optional[ClassificationPayload] = None
    prediction: Optional[PredictionPayload] = None
    allocation: Optional[AllocationPayload] = None
    coordination: Optional[CoordinationPayload] = None
    rescue: Optional[RescuePayload] = None
    delay: Optional[DelayPayload] = None
    evacuate: Optional[EvacuationPayload] = None

    class Config:
        use_enum_values = True


class CrisisObservation(BaseModel):
    """OpenEnv Observation model returned by reset() and step()."""

    threats: List[ThreatInfo] = Field(default_factory=list)
    resources: List[ResourceInfo] = Field(default_factory=list)
    affected_zones: List[AffectedZoneInfo] = Field(default_factory=list)

    time_remaining: int = Field(..., description="Steps left in this episode")
    current_step: int = Field(default=0)
    alerts: List[str] = Field(default_factory=list, description="Human-readable event log")
    episode_id: str = Field(default="", description="Unique ID for this episode")

    resource_budget_remaining: int = Field(default=0, ge=0)
    resource_budget_total: int = Field(default=0, ge=0)
    recent_actions: List[str] = Field(default_factory=list)

    # Action masking metadata for RL agents. Inference clients can ignore this.
    valid_actions: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class CrisisState(BaseModel):
    """OpenEnv State model returned by state()."""

    step_count: int = Field(default=0)
    total_steps: int = Field(default=50)
    episode_id: str = Field(default="")
    difficulty: str = Field(default="medium")

    classification_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_score: float = Field(default=0.0, ge=0.0, le=1.0)
    allocation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coordination_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rescue_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)

    resolved_threats: int = Field(default=0)
    total_threats: int = Field(default=0)
    casualties: int = Field(default=0)
    casualties_prevented: int = Field(default=0)
    total_population_at_risk: int = Field(default=0)
    rescue_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    resource_budget_remaining: int = Field(default=0, ge=0)
    resource_budget_total: int = Field(default=0, ge=0)

    cumulative_reward: float = Field(default=0.0)
    done: bool = Field(default=False)

    task_scores: Dict[str, float] = Field(default_factory=dict)
    efficiency: Dict[str, float] = Field(default_factory=dict)
    progress: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class StepResult(BaseModel):
    """Full return from a single step() call."""

    observation: CrisisObservation
    reward: float = Field(..., description="Internal reward (can be negative)")
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
