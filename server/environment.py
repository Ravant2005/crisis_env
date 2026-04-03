"""
server/environment.py — Core simulation engine for the AI Crisis Response & Rescue Coordination Environment.
Implements: reset(), step(action), state()
Handles: threat lifecycle, resource management, time progression, reward shaping, task grading.
"""

from __future__ import annotations

import uuid
import random
import math
from typing import List, Dict, Tuple, Optional

from models import (
    ActionType, ThreatType, ThreatStatus, ResourceType, ZoneType,
    CrisisAction, CrisisObservation, CrisisState, StepResult,
    ThreatInfo, ResourceInfo, AffectedZoneInfo,
    ClassificationPayload, PredictionPayload,
    AllocationPayload, CoordinationPayload, RescuePayload, EvacuationPayload,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TOTAL_STEPS        = 50          # steps per episode
THREAT_COUNT       = 3           # simultaneous threats at spawn
RESOURCE_COUNT     = 8           # total available resource units
MAX_RESCUE_UNITS   = 5           # max rescue deployments per zone per step
IMPACT_DECAY_RATE  = 1           # TTI decreases by 1 each step
RESOURCE_MATCH_BONUS = 0.3       # bonus effectiveness when resource matches zone
CASCADE_PROBABILITY = 0.4       # probability of cascade threat when not contained

# Zone → preferred resource type mapping (for allocation scoring)
ZONE_RESOURCE_AFFINITY: Dict[ZoneType, List[ResourceType]] = {
    ZoneType.MILITARY: [ResourceType.MILITARY_UNIT, ResourceType.MEDICAL_TEAM],
    ZoneType.MARITIME: [ResourceType.COAST_GUARD, ResourceType.RESCUE_DRONE],
    ZoneType.URBAN:    [ResourceType.SWAT_TEAM, ResourceType.FIRE_BRIGADE, ResourceType.EVACUATION_BUS],
    ZoneType.RURAL:    [ResourceType.FIRE_BRIGADE, ResourceType.MEDICAL_TEAM, ResourceType.RESCUE_DRONE],
}

# Scenario seed templates (threat_type, zone, location, base_population)
THREAT_TEMPLATES: List[Tuple] = [
    (ThreatType.AIRSTRIKE,    ZoneType.MILITARY, "Military Base Alpha",      50),
    (ThreatType.SHIP_ATTACK,  ZoneType.MARITIME, "Naval Port Sector 7",     200),
    (ThreatType.DRONE_THREAT, ZoneType.URBAN,    "Downtown Shopping Mall", 1000),
    (ThreatType.EXPLOSION,    ZoneType.URBAN,    "Central Train Station",   800),
    (ThreatType.FLOOD,        ZoneType.RURAL,    "River Valley District",   400),
    (ThreatType.FIRE,         ZoneType.URBAN,    "Industrial Complex East",  300),
]

RESOURCE_TEMPLATES: List[Tuple] = [
    (ResourceType.MILITARY_UNIT,  ZoneType.MILITARY, 0.90),
    (ResourceType.COAST_GUARD,    ZoneType.MARITIME, 0.88),
    (ResourceType.SWAT_TEAM,      ZoneType.URBAN,    0.85),
    (ResourceType.FIRE_BRIGADE,   ZoneType.URBAN,    0.80),
    (ResourceType.MEDICAL_TEAM,   ZoneType.MILITARY, 0.75),
    (ResourceType.RESCUE_DRONE,   ZoneType.MARITIME, 0.70),
    (ResourceType.EVACUATION_BUS, ZoneType.URBAN,    0.65),
    (ResourceType.MEDICAL_TEAM,   ZoneType.RURAL,    0.78),
]


# ─────────────────────────────────────────────
# GRADER IMPORTS (inline — avoids circular deps)
# ─────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class CrisisEnvironment:
    """
    Full simulation environment for AI Crisis Response & Rescue Coordination.

    Lifecycle:
        env = CrisisEnvironment()
        obs = env.reset()
        while not done:
            action = agent.decide(obs)
            result = env.step(action)
            obs, reward, done = result.observation, result.reward, result.done
        metrics = env.state()
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng  = random.Random(seed)

        # Episode state
        self._episode_id:   str = ""
        self._step_count:   int = 0
        self._done:         bool = False

        # Simulation objects
        self._threats:       List[ThreatInfo]       = []
        self._resources:     List[ResourceInfo]     = []
        self._affected_zones: List[AffectedZoneInfo] = []

        # Running metrics
        self._cumulative_reward:     float = 0.0
        self._casualties:            int   = 0
        self._casualties_prevented:  int   = 0
        self._total_population:      int   = 0
        self._difficulty:           str = "medium"
        self._cascade_threshold:     float = 0.6

        # Per-task grading accumulators
        # Classification
        self._classify_attempts:    int   = 0
        self._classify_full_correct: int   = 0  # type + severity both correct
        self._classify_partial_correct: int = 0  # only type correct

        # Prediction
        self._predict_attempts:   int   = 0
        self._predict_errors:     List[float] = []

        # Allocation
        self._alloc_attempts:     int   = 0
        self._alloc_scores:       List[float] = []

        # Coordination
        self._coord_attempts:     int   = 0
        self._coord_scores:       List[float] = []

        # Rescue
        self._rescue_total_victims:  int = 0
        self._rescue_saved:          int = 0
        self._rescue_steps:          List[int] = []   # step number of each rescue action

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, difficulty: str = "medium", **kwargs) -> CrisisObservation:
        """
        Reset the environment to a fresh randomised scenario.
        Returns the initial observation.
        
        Args:
            seed: Optional seed for reproducibility (default 42 if None)
            difficulty: "easy", "medium", or "hard" - controls number of threats and cascade probability
        """
        if seed is not None:
            self._seed = seed
        if self._seed is None:
            self._seed = 42
        self._rng = random.Random(self._seed)

        # Map difficulty to threat count and cascade threshold
        difficulty_map = {
            "easy": (2, 0.8),
            "medium": (3, 0.6),
            "hard": (5, 0.4),
        }
        threat_count, cascade_threshold = difficulty_map.get(difficulty, (3, 0.6))
        self._difficulty = difficulty
        self._cascade_threshold = cascade_threshold
        
        self._episode_id  = str(uuid.uuid4())[:8]
        self._step_count  = 0
        self._done        = False

        self._cumulative_reward    = 0.0
        self._casualties           = 0
        self._casualties_prevented = 0

        self._classify_attempts = 0
        self._classify_full_correct = 0
        self._classify_partial_correct = 0
        self._predict_attempts  = 0
        self._predict_errors    = []
        self._alloc_attempts    = 0
        self._alloc_scores      = []
        self._coord_attempts    = 0
        self._coord_scores      = []
        self._rescue_total_victims = 0
        self._rescue_saved         = 0
        self._rescue_steps         = []

        # Generate scenario
        self._threats        = self._generate_threats(threat_count)
        self._resources      = self._generate_resources()
        self._affected_zones = []
        self._total_population = sum(t.population_at_risk for t in self._threats)

        alerts = [
            f"[EPISODE {self._episode_id}] INITIATED — {len(self._threats)} simultaneous threats detected (difficulty={difficulty}).",
            f"Available resources: {len(self._resources)} units across all zones.",
        ]

        return self._build_observation(alerts)

    def step(self, action: CrisisAction) -> StepResult:
        """
        Process one agent action. Advances simulation by one step.
        Returns StepResult(observation, reward, done, info).
        """
        if self._done:
            obs = self._build_observation(["Episode already ended. Call reset()."])
            return StepResult(observation=obs, reward=0.0, done=True, info={})

        reward = 0.0
        alerts: List[str] = []
        info:   Dict      = {}

        # ── 1. Process agent action ──────────────────────────────────────────
        action_reward, action_alerts, action_info = self._process_action(action)
        reward  += action_reward
        alerts  += action_alerts
        info.update(action_info)

        # ── 2. Advance threat lifecycle ─────────────────────────────────────
        impact_reward, impact_alerts = self._advance_threats()
        reward  += impact_reward
        alerts  += impact_alerts

        # ── 3. Time step ─────────────────────────────────────────────────────
        self._step_count += 1

        # ── 4. Step-level survival bonus ────────────────────────────────────
        active_contained = sum(
            1 for t in self._threats
            if t.status in (ThreatStatus.CONTAINED, ThreatStatus.RESOLVED)
        )
        reward += active_contained * 0.1

        # ── 5. Check episode termination ────────────────────────────────────
        all_resolved = all(
            t.status in (ThreatStatus.RESOLVED, ThreatStatus.CONTAINED)
            for t in self._threats
        )
        time_up = self._step_count >= TOTAL_STEPS

        if all_resolved or time_up:
            self._done = True
            terminal_reward, terminal_alerts = self._compute_terminal_reward()
            reward  += terminal_reward
            alerts  += terminal_alerts

        self._cumulative_reward += reward

        return StepResult(
            observation=self._build_observation(alerts),
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> CrisisState:
        """Return current episode performance metrics."""
        c_score  = self._grader_classification()
        p_score  = self._grader_prediction()
        a_score  = self._grader_allocation()
        co_score = self._grader_coordination()
        r_score  = self._grader_rescue()

        final = round((c_score + p_score + a_score + co_score + r_score) / 5.0, 4)

        total_victims = max(self._rescue_total_victims, 1)
        rescue_rate   = _clamp(self._rescue_saved / total_victims)

        return CrisisState(
            step_count=self._step_count,
            total_steps=TOTAL_STEPS,
            episode_id=self._episode_id,
            difficulty=self._difficulty,
            classification_score=c_score,
            prediction_score=p_score,
            allocation_score=a_score,
            coordination_score=co_score,
            rescue_score=r_score,
            final_score=final,
            resolved_threats=sum(
                1 for t in self._threats
                if t.status in (ThreatStatus.RESOLVED, ThreatStatus.CONTAINED)
            ),
            total_threats=len(self._threats),
            casualties=self._casualties,
            casualties_prevented=self._casualties_prevented,
            total_population_at_risk=self._total_population,
            rescue_success_rate=round(rescue_rate, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
        )

    def task_scores(self) -> Dict[str, float]:
        """Return current raw grader scores (0.0-1.0 each). Used for reward alignment."""
        return {
            "classification": self._grader_classification(),
            "prediction":     self._grader_prediction(),
            "allocation":     self._grader_allocation(),
            "coordination":   self._grader_coordination(),
            "rescue":         self._grader_rescue(),
        }

    # ─────────────────────────────────────────
    # ACTION PROCESSING
    # ─────────────────────────────────────────

    def _process_action(
        self, action: CrisisAction
    ) -> Tuple[float, List[str], Dict]:
        """Dispatch action to the correct handler. Returns (reward, alerts, info)."""
        t = action.action_type

        if t == ActionType.CLASSIFY and action.classification:
            return self._handle_classify(action.classification)
        elif t == ActionType.PREDICT and action.prediction:
            return self._handle_predict(action.prediction)
        elif t == ActionType.ALLOCATE and action.allocation:
            return self._handle_allocate(action.allocation)
        elif t == ActionType.COORDINATE and action.coordination:
            return self._handle_coordinate(action.coordination)
        elif t == ActionType.RESCUE and action.rescue:
            return self._handle_rescue(action.rescue)
        elif t == ActionType.EVACUATE and action.evacuate:
            return self._handle_evacuate(action.evacuate)
        else:
            return -0.2, ["[WARN] Invalid or incomplete action received."], {}

    def _handle_classify(
        self, payload: ClassificationPayload
    ) -> Tuple[float, List[str], Dict]:
        threat = self._get_threat(payload.threat_id)
        if threat is None:
            return -0.1, ["[WARN] CLASSIFY: Unknown threat_id."], {}

        self._classify_attempts += 1
        type_correct = (payload.predicted_type == threat.threat_type)
        sev_error    = abs(payload.predicted_severity - threat.severity)
        sev_correct  = sev_error <= 1.5  # within 1.5 severity points

        # Check if zone also matches (for full correctness)
        # The classification action doesn't include zone, so we only check type + severity
        if type_correct and sev_correct:
            self._classify_full_correct += 1
            reward = 1.0
            label  = "CORRECT"
        elif type_correct:
            self._classify_partial_correct += 1
            reward = 0.5
            label  = "PARTIAL (type correct, severity off)"
        else:
            reward = -0.3
            label  = "INCORRECT"

        # Store for observation
        threat.predicted_severity = payload.predicted_severity

        return reward, [
            f"[CLASSIFY] Threat {payload.threat_id} → {label} "
            f"(predicted={payload.predicted_type}/{payload.predicted_severity:.1f}, "
            f"actual={threat.threat_type}/{threat.severity:.1f})"
        ], {"classify_result": label}

    def _handle_predict(
        self, payload: PredictionPayload
    ) -> Tuple[float, List[str], Dict]:
        threat = self._get_threat(payload.threat_id)
        if threat is None:
            return -0.1, ["[WARN] PREDICT: Unknown threat_id."], {}

        self._predict_attempts += 1

        tti_error = abs(payload.predicted_tti - threat.time_to_impact)
        pop_error = abs(payload.predicted_pop - threat.population_at_risk)
        pop_norm  = pop_error / max(threat.population_at_risk, 1)

        # Normalized error: 0 = perfect, 1 = worst
        combined_error = _clamp((tti_error / max(TOTAL_STEPS, 1)) * 0.5 + pop_norm * 0.5)
        self._predict_errors.append(combined_error)

        threat.predicted_tti = payload.predicted_tti
        threat.predicted_pop = payload.predicted_pop

        # Reward proportional to accuracy (clamped to non-negative)
        prediction_reward = max(0.0, 1.0 - combined_error)
        reward = round(prediction_reward, 4)

        return reward, [
            f"[PREDICT] Threat {payload.threat_id} → "
            f"TTI error={tti_error} steps, pop error={pop_error} people, "
            f"combined_err={combined_error:.3f}"
        ], {"predict_error": combined_error}

    def _handle_allocate(
        self, payload: AllocationPayload
    ) -> Tuple[float, List[str], Dict]:
        threat   = self._get_threat(payload.threat_id)
        resource = self._get_resource(payload.resource_id)

        if threat is None or resource is None:
            return -0.1, ["[WARN] ALLOCATE: Invalid threat or resource ID."], {}
        if not resource.is_available:
            return -0.2, [
                f"[WARN] ALLOCATE: Resource {payload.resource_id} already assigned."
            ], {}
        if threat.status != ThreatStatus.ACTIVE:
            return -0.1, [
                f"[WARN] ALLOCATE: Threat {payload.threat_id} is not active."
            ], {}

        # Compute allocation score
        affinity  = ZONE_RESOURCE_AFFINITY.get(threat.zone, [])
        is_match  = resource.resource_type in affinity
        tti_urgency = _clamp(1.0 - (threat.time_to_impact / TOTAL_STEPS))

        base_score  = resource.effectiveness
        match_bonus = RESOURCE_MATCH_BONUS if is_match else 0.0
        score       = _clamp(base_score + match_bonus)

        self._alloc_attempts += 1
        self._alloc_scores.append(score)

        # Apply assignment
        resource.is_available = False
        resource.assigned_to  = payload.threat_id
        threat.assigned_resource = payload.resource_id

        # Reduce time_to_impact for assigned threats (resource buys time)
        reduction = int(3 * resource.effectiveness * (1 + match_bonus))
        threat.time_to_impact = max(0, threat.time_to_impact - reduction)

        # Immediate reward
        reward = score + tti_urgency * 0.3

        return round(reward, 4), [
            f"[ALLOCATE] Resource {resource.resource_type} → Threat {payload.threat_id} "
            f"({'zone match ✓' if is_match else 'zone mismatch ✗'}), "
            f"score={score:.2f}, TTI reduced by {reduction} steps"
        ], {"alloc_score": score, "zone_match": is_match}

    def _handle_coordinate(
        self, payload: CoordinationPayload
    ) -> Tuple[float, List[str], Dict]:
        active_threats = [t for t in self._threats if t.status == ThreatStatus.ACTIVE]
        active_ids     = {t.threat_id for t in active_threats}

        # Filter to valid threat IDs
        valid_order = [tid for tid in payload.priority_order if tid in active_ids]

        if not valid_order:
            return -0.2, ["[WARN] COORDINATE: No valid active threat IDs in priority order."], {}

        self._coord_attempts += 1

        # Compute ideal priority order: severity * population / max(TTI, 1)
        def ideal_score(t: ThreatInfo) -> float:
            return (t.severity * t.population_at_risk) / max(t.time_to_impact, 1)

        ideal_order = sorted(active_threats, key=ideal_score, reverse=True)
        ideal_ids   = [t.threat_id for t in ideal_order]

        # Assign priority ranks
        for rank, tid in enumerate(valid_order):
            t = self._get_threat(tid)
            if t:
                t.priority_rank = rank + 1

        # Kendall-tau-like rank correlation score
        score = self._rank_correlation_score(valid_order, ideal_ids)
        self._coord_scores.append(score)

        reward = score * 1.5  # coordination is critical — amplified reward

        return round(reward, 4), [
            f"[COORDINATE] Priority set: {valid_order}, "
            f"Ideal: {ideal_ids}, "
            f"Coordination score={score:.3f}"
        ], {"coord_score": score}

    def _handle_rescue(
        self, payload: RescuePayload
    ) -> Tuple[float, List[str], Dict]:
        zone = self._get_zone(payload.zone_id)
        if zone is None or not zone.is_active:
            return -0.1, [
                f"[WARN] RESCUE: Zone {payload.zone_id} not found or not yet impacted."
            ], {}

        remaining = zone.total_victims - zone.rescued
        if remaining <= 0:
            return 0.1, [f"[RESCUE] Zone {payload.zone_id}: all victims already rescued."], {}

        units       = min(payload.rescue_units_to_send, MAX_RESCUE_UNITS)
        # Each unit saves ~15 victims, scaled by zone type
        zone_multiplier = {
            ZoneType.URBAN:    1.0,
            ZoneType.MARITIME: 0.8,
            ZoneType.MILITARY: 0.9,
            ZoneType.RURAL:    0.7,
        }.get(zone.zone_type, 1.0)

        saved = min(remaining, int(units * 15 * zone_multiplier))
        zone.rescued               += saved
        zone.rescue_units_deployed += units
        self._rescue_saved         += saved
        self._rescue_steps.append(self._step_count)

        if zone.rescued >= zone.total_victims:
            zone.is_active = False

        # Update rescue total (set once per zone on first rescue)
        # Already tracked on zone activation

        speed_bonus = _clamp(1.0 - (self._step_count / TOTAL_STEPS))
        reward      = (saved / max(zone.total_victims, 1)) + speed_bonus * 0.5

        return round(reward, 4), [
            f"[RESCUE] Zone {payload.zone_id}: "
            f"{saved} victims saved by {units} units "
            f"({zone.rescued}/{zone.total_victims} total). "
            f"Speed bonus={speed_bonus:.2f}"
        ], {"rescued": saved, "zone_id": payload.zone_id}

    def _handle_evacuate(
        self, payload: EvacuationPayload
    ) -> Tuple[float, List[str], Dict]:
        """Handle proactive evacuation action - reduces population before impact."""
        threat = self._get_threat(payload.zone_id)
        zone = self._get_zone(payload.zone_id)
        
        if threat is None and zone is None:
            return -0.1, ["[WARN] EVACUATE: Unknown threat/zone ID."], {}
        
        # Each evacuation unit can move ~20 people
        evac_zone = threat.zone if threat else (zone.zone_type if zone else ZoneType.URBAN)
        evac_multiplier = {
            ZoneType.URBAN:    20,
            ZoneType.MARITIME: 15,
            ZoneType.MILITARY: 25,
            ZoneType.RURAL:    30,
        }.get(evac_zone, 20)
        
        pop_to_evac = threat.population_at_risk if threat else (zone.total_victims if zone else 0)
        evacuated = min(pop_to_evac, payload.evac_units * evac_multiplier)
        
        if threat:
            threat.population_evacuated = evacuated
            threat.population_at_risk = max(0, threat.population_at_risk - evacuated)
        
        if zone:
            zone.evacuated = evacuated
            zone.evacuation_units_deployed += payload.evac_units
        
        reward = (evacuated / max(pop_to_evac, 1)) * 2.0
        
        return round(reward, 4), [
            f"[EVACUATE] Zone {payload.zone_id}: {evacuated} people moved to safety "
            f"by {payload.evac_units} units."
        ], {"evacuated": evacuated, "zone_id": payload.zone_id}

    # ─────────────────────────────────────────
    # THREAT LIFECYCLE
    # ─────────────────────────────────────────

    def _advance_threats(self) -> Tuple[float, List[str]]:
        """Decrement TTI for all active threats. Trigger impact when TTI reaches 0."""
        reward = 0.0
        alerts: List[str] = []

        for threat in self._threats:
            if threat.status != ThreatStatus.ACTIVE:
                continue

            # Decay TTI
            threat.time_to_impact = max(0, threat.time_to_impact - IMPACT_DECAY_RATE)

            if threat.time_to_impact == 0:
                # Impact occurs
                impact_casualties, prevented = self._compute_impact(threat)
                threat.casualties           = impact_casualties
                threat.casualties_prevented = prevented
                threat.status               = ThreatStatus.IMPACTED

                self._casualties           += impact_casualties
                self._casualties_prevented += prevented

                # Create affected zone for rescue
                zone = AffectedZoneInfo(
                    zone_id=threat.threat_id,
                    zone_type=threat.zone,
                    location_name=threat.location_name,
                    total_victims=impact_casualties,
                    is_active=(impact_casualties > 0),
                )
                self._affected_zones.append(zone)
                if impact_casualties > 0:
                    self._rescue_total_victims += impact_casualties

                reward -= impact_casualties * 0.002  # penalty per casualty

                alerts.append(
                    f"[IMPACT] {threat.location_name} — {threat.threat_type} struck! "
                    f"Casualties: {impact_casualties}, Prevented: {prevented}"
                )

                # Free assigned resource
                if threat.assigned_resource is not None:
                    res = self._get_resource(threat.assigned_resource)
                    if res:
                        res.is_available = True
                        res.assigned_to  = None

            elif threat.assigned_resource is not None:
                # Threat with resource assigned → may become contained
                resource = self._get_resource(threat.assigned_resource)
                if resource:
                    contain_prob = resource.effectiveness * 0.4
                    if self._rng.random() < contain_prob:
                        threat.status = ThreatStatus.CONTAINED
                        prevented     = int(threat.population_at_risk * 0.6)
                        threat.casualties_prevented = prevented
                        self._casualties_prevented += prevented

                        resource.is_available = True
                        resource.assigned_to  = None

                        reward += 2.0
                        alerts.append(
                            f"[CONTAINED] {threat.location_name} neutralised! "
                            f"{prevented} casualties prevented."
                        )
                    elif self._rng.random() < self._cascade_threshold:
                        # Threat not contained → spawn cascade secondary threat
                        cascade_type = self._rng.choice([
                            ThreatType.FIRE, ThreatType.EXPLOSION, ThreatType.FLOOD
                        ])
                        cascade_severity = round(threat.severity * 0.6, 1)
                        cascade_pop = int(threat.population_at_risk * 0.4)
                        cascade_tti = self._rng.randint(3, 8)
                        
                        new_threat = ThreatInfo(
                            threat_id=len(self._threats) + 1,
                            threat_type=cascade_type,
                            status=ThreatStatus.ACTIVE,
                            severity=cascade_severity,
                            population_at_risk=max(10, cascade_pop),
                            time_to_impact=cascade_tti,
                            zone=threat.zone,
                            location_name=f"{threat.location_name} (Cascade)",
                        )
                        self._threats.append(new_threat)
                        self._total_population += new_threat.population_at_risk
                        alerts.append(
                            f"[CASCADE] Secondary {cascade_type} at {threat.location_name}! "
                            f"severity={cascade_severity}, population={cascade_pop}, TTI={cascade_tti}"
                        )

        # Mark impacted threats as resolved once zone is fully rescued
        for zone in self._affected_zones:
            if not zone.is_active and zone.rescued >= zone.total_victims:
                threat = self._get_threat(zone.zone_id)
                if threat and threat.status == ThreatStatus.IMPACTED:
                    threat.status = ThreatStatus.RESOLVED
                    reward += 1.0
                    alerts.append(f"[RESOLVED] {threat.location_name} — all victims rescued.")

        return reward, alerts

    def _compute_impact(self, threat: ThreatInfo) -> Tuple[int, int]:
        """
        Compute actual casualties and prevented casualties when a threat impacts.
        Casualties depend on: severity, population, whether resource was assigned.
        """
        base_rate = threat.severity / 10.0  # 0.0–1.0

        # Resource assignment reduces casualty rate
        mitigation = 0.0
        if threat.assigned_resource is not None:
            res = self._get_resource(threat.assigned_resource)
            if res:
                mitigation = res.effectiveness * 0.7

        actual_rate   = _clamp(base_rate - mitigation)
        casualties    = int(threat.population_at_risk * actual_rate)
        prevented     = int(threat.population_at_risk * mitigation * base_rate)

        return casualties, prevented

    # ─────────────────────────────────────────
    # TERMINAL REWARD
    # ─────────────────────────────────────────

    def _compute_terminal_reward(self) -> Tuple[float, List[str]]:
        """Bonus/penalty applied at episode end."""
        alerts: List[str] = []
        reward = 0.0

        total_pop    = max(self._total_population, 1)
        survival_rate = 1.0 - (self._casualties / total_pop)
        reward += survival_rate * 5.0

        resolved = sum(
            1 for t in self._threats
            if t.status in (ThreatStatus.RESOLVED, ThreatStatus.CONTAINED)
        )
        resolve_rate = resolved / max(len(self._threats), 1)
        reward += resolve_rate * 3.0

        rescue_rate = _clamp(self._rescue_saved / max(self._rescue_total_victims, 1))
        reward += rescue_rate * 2.0

        alerts.append(
            f"[TERMINAL] Survival={survival_rate:.2%}, "
            f"Resolved={resolve_rate:.2%}, "
            f"Rescue={rescue_rate:.2%}, "
            f"Terminal reward={reward:.2f}"
        )
        return round(reward, 4), alerts

    # ─────────────────────────────────────────
    # GRADERS (0.0 → 1.0)
    # ─────────────────────────────────────────

    def _grader_classification(self) -> float:
        """Task 1: correct classifications / total attempts."""
        if self._classify_attempts == 0:
            return 0.0
        # Full credit for full correct, 0.5 for partial correct
        raw = (self._classify_full_correct * 1.0 + self._classify_partial_correct * 0.5) / self._classify_attempts
        return round(_clamp(raw), 4)

    def _grader_prediction(self) -> float:
        """Task 2: 1 - mean normalized prediction error."""
        if not self._predict_errors:
            return 0.0
        mean_err = sum(self._predict_errors) / len(self._predict_errors)
        return round(_clamp(1.0 - mean_err), 4)

    def _grader_allocation(self) -> float:
        """Task 3: mean allocation quality score."""
        if not self._alloc_scores:
            return 0.0
        return round(_clamp(sum(self._alloc_scores) / len(self._alloc_scores)), 4)

    def _grader_coordination(self) -> float:
        """Task 4: mean rank correlation score across all coordination actions."""
        if not self._coord_scores:
            return 0.0
        return round(_clamp(sum(self._coord_scores) / len(self._coord_scores)), 4)

    def _grader_rescue(self) -> float:
        """Task 5: composite of lives-saved ratio + speed + resource efficiency."""
        if self._rescue_total_victims == 0:
            # No impacts → perfect score (prevented all)
            return round(_clamp(self._casualties_prevented / max(self._total_population, 1) + 0.5), 4)

        lives_saved_ratio = _clamp(self._rescue_saved / self._rescue_total_victims)

        if self._rescue_steps:
            avg_step     = sum(self._rescue_steps) / len(self._rescue_steps)
            speed_score  = _clamp(1.0 - avg_step / TOTAL_STEPS)
        else:
            speed_score  = 0.0

        total_units  = sum(z.rescue_units_deployed for z in self._affected_zones)
        total_saved  = max(self._rescue_saved, 1)
        efficiency   = _clamp(total_saved / max(total_units * 15, 1))

        composite = (lives_saved_ratio + speed_score + efficiency) / 3.0
        return round(_clamp(composite), 4)

    # ─────────────────────────────────────────
    # GENERATION HELPERS
    # ─────────────────────────────────────────

    def _generate_threats(self, threat_count: int = 3) -> List[ThreatInfo]:
        """Generate threat_count randomised threats from templates."""
        chosen    = self._rng.sample(THREAT_TEMPLATES, threat_count)
        threats   = []
        for i, (ttype, zone, location, base_pop) in enumerate(chosen):
            severity   = round(self._rng.uniform(4.0, 10.0), 1)
            pop_jitter = self._rng.randint(-int(base_pop * 0.2), int(base_pop * 0.2))
            population = max(10, base_pop + pop_jitter)
            tti        = self._rng.randint(5, 20)   # 5–20 steps before impact

            threats.append(ThreatInfo(
                threat_id=i + 1,
                threat_type=ttype,
                status=ThreatStatus.ACTIVE,
                severity=severity,
                population_at_risk=population,
                time_to_impact=tti,
                zone=zone,
                location_name=location,
            ))
        return threats

    def _generate_resources(self) -> List[ResourceInfo]:
        """Generate a fixed pool of resource units."""
        resources = []
        for i, (rtype, zone, eff) in enumerate(RESOURCE_TEMPLATES):
            eff_jitter = round(self._rng.uniform(-0.05, 0.05), 3)
            resources.append(ResourceInfo(
                resource_id=i + 1,
                resource_type=rtype,
                is_available=True,
                assigned_to=None,
                effectiveness=round(_clamp(eff + eff_jitter), 3),
                location_zone=zone,
            ))
        return resources

    # ─────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────

    def _build_observation(self, alerts: List[str]) -> CrisisObservation:
        enriched_threats = []
        for t in self._threats:
            tti = max(t.time_to_impact, 1)
            priority_score = (t.severity * t.population_at_risk) / tti
            risk_level = "high" if priority_score > 500 else "medium" if priority_score > 100 else "low"
            if tti <= 2 and t.population_at_risk > 1000:
                recommended_action_hint = "evacuate"
            elif t.severity >= 4:
                recommended_action_hint = "allocate_resources"
            else:
                recommended_action_hint = "classify_and_monitor"
            
            t.priority_score = priority_score
            t.risk_level = risk_level
            t.recommended_action_hint = recommended_action_hint
            enriched_threats.append(t)
        
        return CrisisObservation(
            threats=enriched_threats,
            resources=list(self._resources),
            affected_zones=list(self._affected_zones),
            time_remaining=max(0, TOTAL_STEPS - self._step_count),
            current_step=self._step_count,
            alerts=alerts,
            episode_id=self._episode_id,
        )

    def _get_threat(self, threat_id: int) -> Optional[ThreatInfo]:
        for t in self._threats:
            if t.threat_id == threat_id:
                return t
        return None

    def _get_resource(self, resource_id: int) -> Optional[ResourceInfo]:
        for r in self._resources:
            if r.resource_id == resource_id:
                return r
        return None

    def _get_zone(self, zone_id: int) -> Optional[AffectedZoneInfo]:
        for z in self._affected_zones:
            if z.zone_id == zone_id:
                return z
        return None

    @staticmethod
    def _rank_correlation_score(agent_order: List[int], ideal_order: List[int]) -> float:
        """
        Compute a normalised rank-correlation score between agent's ordering
        and the ideal ordering. Score ∈ [0.0, 1.0].
        Uses position-weighted agreement: top positions matter more.
        """
        if not ideal_order:
            return 0.0

        n           = len(ideal_order)
        ideal_rank  = {tid: rank for rank, tid in enumerate(ideal_order)}
        agent_rank  = {tid: rank for rank, tid in enumerate(agent_order) if tid in ideal_rank}

        if not agent_rank:
            return 0.0

        total_weight = 0.0
        weighted_err = 0.0

        for tid, a_rank in agent_rank.items():
            i_rank = ideal_rank[tid]
            weight = 1.0 / (i_rank + 1)   # higher weight for top priorities
            err    = abs(a_rank - i_rank) / max(n - 1, 1)
            total_weight += weight
            weighted_err += weight * err

        if total_weight == 0:
            return 0.0

        return round(_clamp(1.0 - weighted_err / total_weight), 4)
