"""
models.py — Typed Pydantic models for the AI Crisis Response & Rescue Coordination Environment.
Implements the full OpenEnv spec: Action, Observation, State, and supporting data models.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class ActionType(str, Enum):
    CLASSIFY   = "classify"      # Task 1 — identify threat type + severity
    PREDICT    = "predict"       # Task 2 — predict time-to-impact + population affected
    ALLOCATE   = "allocate"      # Task 3 — assign a resource to a threat
    COORDINATE = "coordinate"   # Task 4 — set priority ordering across all active threats
    RESCUE     = "rescue"        # Task 5 — deploy rescue units post-impact
    EVACUATE   = "evacuate"     # Proactive evacuation before impact


class ThreatType(str, Enum):
    AIRSTRIKE    = "airstrike"
    SHIP_ATTACK  = "ship_attack"
    DRONE_THREAT = "drone_threat"
    EXPLOSION    = "explosion"
    FLOOD        = "flood"
    FIRE         = "fire"


class ThreatStatus(str, Enum):
    ACTIVE    = "active"
    IMPACTED  = "impacted"
    CONTAINED = "contained"
    RESOLVED  = "resolved"


class ResourceType(str, Enum):
    MILITARY_UNIT  = "military_unit"
    COAST_GUARD    = "coast_guard"
    SWAT_TEAM      = "swat_team"
    FIRE_BRIGADE   = "fire_brigade"
    MEDICAL_TEAM   = "medical_team"
    RESCUE_DRONE   = "rescue_drone"
    EVACUATION_BUS = "evacuation_bus"


class ZoneType(str, Enum):
    MILITARY = "military"
    MARITIME = "maritime"
    URBAN    = "urban"
    RURAL    = "rural"


# ─────────────────────────────────────────────
# SUPPORTING SUB-MODELS
# ─────────────────────────────────────────────

class ThreatInfo(BaseModel):
    """Represents a single active or resolved threat in the environment."""
    threat_id:          int               = Field(..., description="Unique threat identifier")
    threat_type:        ThreatType        = Field(..., description="Category of threat")
    status:             ThreatStatus      = Field(default=ThreatStatus.ACTIVE)
    severity:           float             = Field(..., ge=0.0, le=10.0, description="Threat severity score 0–10")
    population_at_risk: int               = Field(..., ge=0,  description="Number of civilians/personnel at risk")
    time_to_impact:     int               = Field(..., ge=0,  description="Steps remaining until impact occurs")
    zone:               ZoneType          = Field(..., description="Geographic zone type")
    location_name:      str               = Field(..., description="Human-readable location label")
    predicted_severity: Optional[float]   = Field(default=None, description="Agent's predicted severity (Task 1)")
    predicted_tti:      Optional[int]     = Field(default=None, description="Agent's predicted time-to-impact (Task 2)")
    predicted_pop:      Optional[int]     = Field(default=None, description="Agent's predicted population (Task 2)")
    assigned_resource:  Optional[int]     = Field(default=None, description="Resource ID assigned to this threat")
    priority_rank:      Optional[int]     = Field(default=None, description="Agent's assigned priority rank (Task 4)")
    casualties:         int               = Field(default=0,  description="Casualties incurred after impact")
    casualties_prevented: int             = Field(default=0,  description="Casualties prevented by timely response")
    population_evacuated: int             = Field(default=0,  description="Population evacuated before impact")
    priority_score:     float             = Field(default=0.0, description="Computed: severity * population / max(TTI, 1)")
    risk_level:         str               = Field(default="low", description="Computed: high/medium/low based on priority_score")
    recommended_action_hint: str          = Field(default="classify_and_monitor", description="Computed action recommendation")


class ResourceInfo(BaseModel):
    """Represents a single deployable resource unit."""
    resource_id:    int          = Field(..., description="Unique resource identifier")
    resource_type:  ResourceType = Field(..., description="Type of resource unit")
    is_available:   bool         = Field(default=True, description="Whether resource is currently free")
    assigned_to:    Optional[int]= Field(default=None, description="Threat ID this resource is assigned to")
    effectiveness:  float        = Field(..., ge=0.0, le=1.0, description="Effectiveness multiplier for this unit")
    location_zone:  ZoneType     = Field(..., description="Zone this resource is stationed in")


class AffectedZoneInfo(BaseModel):
    """Post-impact zone requiring rescue operations."""
    zone_id:            int      = Field(..., description="Unique zone identifier")
    zone_type:          ZoneType = Field(...)
    location_name:      str      = Field(...)
    total_victims:      int      = Field(default=0, description="Total victims needing rescue")
    rescued:            int      = Field(default=0, description="Victims already rescued")
    rescue_units_deployed: int   = Field(default=0)
    is_active:          bool     = Field(default=False, description="True if impact occurred here")
    evacuated:          int      = Field(default=0, description="Population evacuated before impact")
    evacuation_units_deployed: int = Field(default=0, description="Units used for evacuation")


class ClassificationPayload(BaseModel):
    """Payload for CLASSIFY action."""
    threat_id:          int   = Field(..., description="Which threat to classify")
    predicted_type:     ThreatType = Field(..., description="Agent's predicted threat type")
    predicted_severity: float = Field(..., ge=0.0, le=10.0, description="Agent's predicted severity")


class PredictionPayload(BaseModel):
    """Payload for PREDICT action."""
    threat_id:      int   = Field(..., description="Which threat to predict impact for")
    predicted_tti:  int   = Field(..., ge=0, description="Predicted steps to impact")
    predicted_pop:  int   = Field(..., ge=0, description="Predicted population affected")


class AllocationPayload(BaseModel):
    """Payload for ALLOCATE action."""
    threat_id:   int = Field(..., description="Target threat")
    resource_id: int = Field(..., description="Resource to assign")


class CoordinationPayload(BaseModel):
    """Payload for COORDINATE action — sets priority ranking across all active threats."""
    priority_order: List[int] = Field(
        ..., description="Ordered list of threat_ids from highest to lowest priority"
    )


class RescuePayload(BaseModel):
    """Payload for RESCUE action — deploys rescue units into an affected zone."""
    zone_id:              int = Field(..., description="Affected zone to rescue")
    rescue_units_to_send: int = Field(..., ge=1, description="Number of rescue units to dispatch")


class EvacuationPayload(BaseModel):
    """Payload for EVACUATE action — proactive evacuation before impact."""
    zone_id:         int = Field(..., description="Zone to evacuate")
    evac_units:      int = Field(..., ge=1, description="Number of evacuation units to deploy")
    population_move: int = Field(default=0, description="Estimated population moved to safety")


# ─────────────────────────────────────────────
# CORE OpenEnv MODELS
# ─────────────────────────────────────────────

class CrisisAction(BaseModel):
    """
    OpenEnv Action model.
    The agent submits one action per step.
    """
    action_type: ActionType = Field(..., description="Which task the agent is performing")

    # Only ONE of these payloads will be populated per action, matching action_type
    classification: Optional[ClassificationPayload] = None
    prediction:     Optional[PredictionPayload]     = None
    allocation:     Optional[AllocationPayload]     = None
    coordination:   Optional[CoordinationPayload]   = None
    rescue:         Optional[RescuePayload]         = None
    evacuate:       Optional[EvacuationPayload]    = None

    class Config:
        use_enum_values = True


class CrisisObservation(BaseModel):
    """
    OpenEnv Observation model.
    Returned by reset() and step() — everything the agent can see.
    """
    threats:        List[ThreatInfo]       = Field(default_factory=list)
    resources:      List[ResourceInfo]     = Field(default_factory=list)
    affected_zones: List[AffectedZoneInfo] = Field(default_factory=list)
    time_remaining: int                    = Field(..., description="Steps left in this episode")
    current_step:   int                    = Field(default=0)
    alerts:         List[str]              = Field(default_factory=list, description="Human-readable event log")
    episode_id:     str                    = Field(default="", description="Unique ID for this episode")

    class Config:
        use_enum_values = True


class CrisisState(BaseModel):
    """
    OpenEnv State model.
    Returned by state() — full performance metrics for the current episode.
    """
    step_count:           int   = Field(default=0)
    total_steps:          int   = Field(default=50)
    episode_id:           str   = Field(default="")
    difficulty:           str   = Field(default="medium", description="Difficulty level set for this episode")

    # Task-level scores (0.0–1.0)
    classification_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_score:     float = Field(default=0.0, ge=0.0, le=1.0)
    allocation_score:     float = Field(default=0.0, ge=0.0, le=1.0)
    coordination_score:   float = Field(default=0.0, ge=0.0, le=1.0)
    rescue_score:         float = Field(default=0.0, ge=0.0, le=1.0)
    final_score:          float = Field(default=0.0, ge=0.0, le=1.0)

    # Outcome counters
    resolved_threats:      int  = Field(default=0)
    total_threats:         int  = Field(default=0)
    casualties:            int  = Field(default=0)
    casualties_prevented:  int  = Field(default=0)
    total_population_at_risk: int = Field(default=0)
    rescue_success_rate:   float = Field(default=0.0, ge=0.0, le=1.0)

    # Running reward
    cumulative_reward:    float = Field(default=0.0)
    done:                 bool  = Field(default=False)

    class Config:
        use_enum_values = True


class StepResult(BaseModel):
    """Full return from a single step() call."""
    observation: CrisisObservation
    reward:      float = Field(..., description="Internal reward (can be negative)")
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
