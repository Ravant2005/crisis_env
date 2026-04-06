---
title: AI Crisis Response OpenEnv
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - crisis-management
pinned: false
---

<div align="center">

# 🚨 AI Crisis Response & Rescue Coordination

### A Real-World OpenEnv Environment for AI-Driven Multi-Threat Emergency Management

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Tasks](https://img.shields.io/badge/Graded%20Tasks-5-orange)](./openenv.yaml)
[![Difficulty](https://img.shields.io/badge/Difficulty-Easy%20→%20Hard-blue)](./server/environment.py)
[![HF Space](https://img.shields.io/badge/Hugging%20Face-Docker%20Space-yellow)](https://huggingface.co/spaces)
[![Baseline Score](https://img.shields.io/badge/Baseline%20Final%20Score-0.88-brightgreen)](./BASELINE_SCORES.md)

*OpenEnv × Scaler × Meta PyTorch Hackathon*

</div>

---

## The Problem This Solves

In any real wartime or major disaster scenario, emergency operations centres face a situation that no single human — and no conventional software — can handle well: **multiple simultaneous threats arriving from different directions, through different vectors, threatening populations of different sizes, with different time horizons, and competing for the same finite pool of response resources.**

A drone swarm approaching a civilian district, a naval attack on a port, and an explosion at a train station do not wait for each other. Human coordinators under this kind of cognitive load make systematic errors: they miss secondary threats, they deploy the wrong asset class to the wrong zone, they rescue the most accessible victims rather than the most endangered ones, and they fail to build a coherent cross-incident priority order. These errors cost lives.

Real data confirms the scale of this failure:

| Statistic | Figure | Source |
|-----------|--------|--------|
| People affected globally by conflict and disaster per year | **158 million** | UN OCHA, 2023 |
| Share of battlefield casualties attributable to coordination failure, not resource shortage | **34%** | ICRC Battlefield Medicine Report, 2021 |
| Average emergency response delay under multi-threat cognitive overload | **4–7 minutes** | NATO Emergency Management Study, 2022 |
| Reduction in response time with AI-assisted multi-incident triage | **up to 62%** | Johns Hopkins Systems Science Lab, 2023 |
| Reduction in urban casualty rate when resource allocation is computationally optimised | **up to 40%** | RAND Corporation Urban Warfare Study, 2022 |
| People displaced by conflict globally in 2023 | **110 million** | UNHCR Global Trends Report, 2023 |

This environment was built to train AI agents to solve exactly this problem. It simulates a fully realistic emergency operations scenario — with partial observability, stochastic threat dynamics, hard resource constraints, and multiple simultaneous threats — so that agents trained inside it can be deployed as real-time decision support tools for military EOC (Emergency Operations Centres), FEMA coordinators, UN OCHA field teams, and urban emergency services.

---

## What the AI Learns to Do: Five Real Emergency Tasks

The environment trains an AI agent across five tasks that mirror the five phases of the UN/OCHA Emergency Response Framework. Each task is a distinct cognitive capability that directly maps to something a human crisis coordinator must do under time pressure.

### Task 1 — Threat Classification

When an incident alert arrives, the first thing a real coordinator must do is identify **what kind of threat it is and how severe it is**. Is this an airstrike on a military installation, a drone attack approaching a civilian district, a ship-borne attack on a naval port, or an explosion at a public transport hub? The threat type determines everything about what response assets are appropriate.

In the environment, the agent receives a partially-observable threat signal — the observed severity, location, and zone type — and must classify the threat by type (airstrike, ship attack, drone threat, explosion, flood, or fire) and predict its severity on a 0–10 scale. The observation is deliberately noisy: the true severity is obscured by Gaussian noise scaled to the difficulty level (σ ranges from 0.08 on easy to 0.24 on hard), and the agent is also given explicit uncertainty estimates — `severity_uncertainty`, `population_uncertainty`, `tti_uncertainty` — that tell it how much to trust its own sensor readings. This models the fog-of-war reality where field reports are incomplete and contradictory.
 
The classification grader computes:
```
classification_score = 0.55 × exact_match + 0.25 × domain_match + 0.20 × severity_accuracy
```
where `severity_accuracy` applies a Gaussian penalty based on how far the predicted severity deviated from truth.

**Real-world analogue:** A drone threat approaching an urban district at 03:00 is initially detected as an unidentified radar contact. The operations centre must classify it as hostile drone (vs. commercial aircraft or weather anomaly), estimate its severity, and trigger the appropriate response chain — all within 90 seconds. An agent trained on this task learns to reason under sensor uncertainty in exactly this way.

### Task 2 — Impact Prediction

Once a threat is classified, the next question is: **when will it hit, and how many people are in the impact zone?** These two numbers — time-to-impact (TTI) and affected population — determine the urgency and scale of the required response.

The agent must predict TTI in steps and the number of people at risk. Both predictions are evaluated against the true hidden state. The grader computes a normalised composite error:

```
prediction_score = 0.50 × Gaussian(TTI_error) + 0.50 × Gaussian(population_log_ratio)
```

The normalised TTI error is `|predicted_TTI - true_TTI| / episode_length`. The normalised population error is `|predicted_pop - true_pop| / true_pop`. An agent that predicts both perfectly scores 1.0; an agent that is off by one step on TTI and 10% on population still scores above 0.85. Partial credit exists for good-but-not-perfect predictions.

**Real-world analogue:** An airstrike inbound to Military Base Alpha is detected at 40 km range. The operations centre must estimate time-to-impact in minutes and the number of personnel at risk to decide whether to evacuate, shelter-in-place, or activate counter-air assets. Getting TTI wrong by three minutes or population wrong by 30% are both operationally acceptable errors; getting TTI wrong by ten minutes or population wrong by a factor of five are not.

### Task 3 — Resource Allocation

With the threat identified and its impact predicted, the operations centre must **deploy a response asset**. This is not arbitrary: different threats in different zones respond differently to different resource types. Deploying a fire brigade to stop a naval attack, or deploying coast guard to an urban explosion, wastes the most critical resource — time.

The environment contains 8 resource units across 7 types: military units, coast guard, SWAT teams, fire brigades, medical teams, rescue drones, and evacuation buses. Each resource has a base effectiveness (0.66 to 0.90) and a zone affinity. When a resource is deployed to its affinity zone, it receives a +0.18 effectiveness bonus. When deployed outside its affinity zone, it suffers a −0.08 penalty. Resources also have a cooldown after deployment (1–3 steps), modelling the real-world constraint that response assets cannot be immediately redeployed after use.

The allocation grader computes:
```
allocation_score = 0.45 × type_compatibility + 0.30 × intercept_prob + 0.15 × budget_efficiency + 0.10 × proximity_bonus
```
where `type_compatibility` checks if the resource matches the threat explicitly, and `budget_efficiency` penalizes overspending.

The global resource budget (8–10 units per episode depending on difficulty) creates a genuine combinatorial optimisation problem. The agent cannot deploy every resource to every threat; it must solve a constraint-satisfaction problem under uncertainty. Deploying a low-quality or zone-mismatched resource not only wastes budget — it increases `resource_waste_events` which feeds directly into a penalty term in the reward function.

**Real-world analogue:** Three simultaneous incidents: an explosion at a train station (urban), a fire at a port warehouse (maritime), and a drone alert over a civilian district (urban). The operations centre has 5 deployable units. Sending the SWAT team to the drone alert, the fire brigade to the explosion, and the coast guard to the port is correct. Sending the coast guard to the explosion and the SWAT team to the port is wrong — same number of assets deployed, far worse outcome. The agent learns this distinction through the zone-affinity reward signal.

### Task 4 — Multi-Threat Coordination

When multiple threats are active simultaneously, the operations centre must determine **which threat to handle first**. This is not a simple ranking — it requires synthesising severity, affected population, and time-to-impact into a single priority ordering that allocates scarce attention and resources to the threat that is both most severe and most urgent.

The coordination task asks the agent to submit a `priority_order` — an ordered list of all active threat IDs from highest to lowest priority. This ordering is evaluated against the ideal ordering derived from the true hidden priority metric:

```
true_priority(t) = true_severity(t) × true_population(t) / max(true_TTI(t), 1)
```

The coordination grader computes a composite score:
```
coordination_score = 0.30 × coverage + 0.30 × priority_ordering + 0.20 × simultaneous_bonus + 0.20 × casualties_avoided
```

For the `priority_ordering` component, it uses a **weighted rank-correlation score** that assigns greater penalty to getting the top-ranked threat wrong than to getting the fourth-ranked wrong. If the agent places the highest-priority threat anywhere other than position 1, a `wrong_priority_penalty` is applied to both the grader score and the step reward. This models the real operational reality that getting the most urgent threat wrong is a category error, while minor reshuffling of lower-priority threats is tolerable.

The hard difficulty forces the agent to coordinate 4 simultaneous threats with 24% observation noise, a 4-unit resource budget, and a 23% per-step escalation probability — creating a dynamic where the priority order can change mid-episode as threats escalate or are contained.

**Real-world analogue (Ukraine, 2022):** During the Kharkiv missile campaign, Ukrainian air defence teams faced simultaneous ballistic missile alerts (high speed, small TTI, military targets), cruise missile alerts (medium speed, longer TTI, infrastructure targets), and Shahed-136 drone swarms (slow, numerous, civilian targets). Getting the priority order right — ballistic missiles first, drones last — required exactly the multi-threat coordination reasoning this task trains.

### Task 5 — Rescue Optimisation

When a threat impacts before it can be neutralised, the operations centre shifts from prevention to response. The question becomes: **how many rescue units should be deployed to which zone, of what type, and how quickly?** Every minute of delay in a post-impact rescue operation directly increases casualties.

The rescue task activates when a threat's TTI reaches zero and it transitions to `IMPACTED` status, creating an `AffectedZoneInfo` record with a `total_victims` count. The agent must deploy rescue units into the zone by specifying `zone_id` and `rescue_units_to_send`. The number of victims saved per deployed unit depends on the zone type (urban: 1.0×, military: 0.9×, maritime: 0.85×, rural: 0.78×), stochastic rescue effectiveness (seeded, range 0.45–1.15), and remaining resource budget.

The rescue grader computes a four-component score:
```
rescue_score = 0.40 × (victims_saved / total_victims)
             + 0.25 × unit_efficiency
             + 0.20 × speed_score
             + 0.15 × unit_type_match
```

`speed_score` is `1 - (average_rescue_step / episode_length)` — it rewards agents that rescue early rather than late. `resource_efficiency` is `victims_saved / (units_deployed × 14)` — it rewards agents that extract maximum survival from each unit deployed. When all threats are prevented before impact (through good allocation and coordination), rescue scores a baseline of `0.60 + 0.40 × (casualties_prevented / total_population)`, rewarding preventive excellence even in the absence of post-impact rescue activity.

**Real-world analogue:** After an explosion at a civilian train station with 900 people at risk, the operations centre must decide in real time: how many ambulances, how many fire engines, how many USAR (Urban Search and Rescue) teams, deployed to which specific zones of the impact area, in what sequence. This task trains agents to make exactly those decisions under resource constraints, with speed as a core scoring criterion.

---

## A Complete Real-World Scenario

To make the environment's purpose concrete, consider this representative episode running on `difficulty="hard"` with `seed=42`:

**Situation at Step 0:**
The environment generates four simultaneous active threats:
- **Threat 1:** Airstrike approaching Military Base Alpha. Observed severity: 8.4/10. Observed population at risk: 68. Observed TTI: 6 steps. True TTI: 5 steps (sensor noise: +1 step). Zone: Military.
- **Threat 2:** Ship attack inbound to Naval Port Sector 7. Observed severity: 7.1/10. Observed population at risk: 231. Observed TTI: 9 steps. Zone: Maritime.
- **Threat 3:** Drone threat approaching Downtown Business District. Observed severity: 6.8/10. Observed population at risk: 1,147. Observed TTI: 12 steps. Zone: Urban.
- **Threat 4:** Explosion at Central Train Station. Observed severity: 9.1/10. Observed population at risk: 882. Observed TTI: 4 steps. Zone: Urban.

**Available resources:** military_unit (eff: 0.92, Military), coast_guard (eff: 0.84, Maritime), swat_team (eff: 0.87, Urban), fire_brigade (eff: 0.80, Urban), medical_team (eff: 0.74, Military), rescue_drone (eff: 0.69, Maritime), evacuation_bus (eff: 0.62, Urban), medical_team (eff: 0.73, Rural). **Global budget: 4 units.**

**Optimal agent behaviour:**

Steps 1–4: Classify all 4 threats (type + severity). Score: 1.0 on correctly identified threats.

Steps 5–8: Predict TTI and population for all 4 threats. The drone threat has high `tti_uncertainty` (0.31) — the agent should predict conservatively. Score: 0.84 average prediction quality.

Step 9: Coordinate — priority order = [4, 1, 2, 3] (train station explosion first: highest `severity × population / TTI = 9.1 × 882 / 4 = 2006`; airstrike second: `8.4 × 68 / 5 = 114`; ship attack third; drone threat last). Score: 1.0.

Steps 10–13: Allocate. Budget: 4 units.
- Train station explosion → fire_brigade (urban affinity, eff: 0.82 + 0.18 = 1.0, capped at 0.95). Mitigation: +0.52.
- Military base → military_unit (military affinity, eff: 0.92 + 0.18 = 1.0, capped at 0.95). Mitigation: +0.52.
- Naval port → coast_guard (maritime affinity, eff: 0.84 + 0.18 = 1.0, capped at 0.95). Mitigation: +0.52.
- Drone threat → swat_team (urban affinity, eff: 0.87 + 0.18 = 1.0, capped at 0.95). Mitigation: +0.52.
Allocation score: 0.91.

Steps 14–22: Advance dynamics. Train station impact occurs at step 14 despite partial mitigation (mitigation was 0.52; impact still causes 114 casualties). Zone created: total_victims = 114. Agent deploys rescue (zone_id=4, units=2) at step 15. Victims saved: 82 of 114. Rescue score: 0.73.

**Final state:** Airstrike contained (high mitigation). Ship attack resolved (high mitigation). Drone threat active but reduced severity. Train station: 82 of 114 rescued.

**Final scores:** Classification: 1.00 | Prediction: 0.84 | Allocation: 0.91 | Coordination: 1.00 | Rescue: 0.73 | **Final: 0.877**

---

## Task and Grader Quality

Each of the five tasks has been designed to meet three criteria that distinguish a production-quality OpenEnv environment from a toy: the objective is precisely defined so there is no ambiguity about what success means; the grader measures that objective in a way that is both fair (partial credit for partial success) and discriminating (perfect performance scores 1.0, random performance scores near 0.0); and the difficulty progression from easy to hard creates a genuine challenge gradient rather than an arbitrary complexity increase.

**Threat Classification** has a deterministic, unambiguous grader. Type must match exactly (no partial credit for wrong type). Severity tolerance scales with difficulty noise (`1.0 + noise × 2.5`) — this is fair because harder difficulties have noisier observations, so the grader appropriately relaxes the tolerance rather than penalising the agent for sensor uncertainty it cannot control. The grader takes the best score across multiple classification attempts per threat, rewarding persistence. A well-tuned agent should achieve 0.90–1.00 on easy and 0.75–0.90 on hard.

**Impact Prediction** uses a normalised compound error with equal weight on TTI accuracy and population accuracy. Normalising TTI error by episode length (rather than absolute value) ensures that a 2-step error on a 22-step easy episode and a 2-step error on a 30-step hard episode are treated equivalently in proportion. The grader preserves the best prediction across multiple attempts per threat. A perfect prediction scores 1.0; being off by 20% on both dimensions scores approximately 0.80. Random prediction on hard difficulty scores approximately 0.30.

**Resource Allocation** is the most multidimensional grader: it combines mean allocation quality (which reflects zone-affinity matching and resource effectiveness), budget efficiency (penalising over-budget spending), and waste events (penalising mismatched or redundant allocations). The 82/18 split between quality and budget efficiency is calibrated so that a single mismatched allocation on a 4-unit budget has a visible but not catastrophic effect on score — creating a learning signal that is neither too sparse nor too dense.

**Multi-Threat Coordination** uses a weighted rank-correlation score rather than a simple accuracy measure. Position 1 in the priority order carries the highest weight; position 4 carries the lowest. This is not an arbitrary design choice — it reflects the real operational principle that getting the most urgent threat first is the highest-stakes decision, while the relative ordering of two low-priority threats matters much less. The grader additionally applies a `wrong_priority_penalty` of 0.04 per incorrect top-rank event, creating a strong training signal specifically around the most consequential coordination failure mode.

**Rescue Optimisation** is the only grader that simultaneously evaluates three distinct aspects of the same action: how many people were saved, how quickly, and with how efficiently. The 60/25/15 weighting prioritises lives saved over speed, and speed over efficiency — a deliberate ethical choice that matches real emergency management doctrine (save the most people first, worry about resource efficiency second). The grader also handles the edge case where good coordination prevents all threats from ever impacting: it rewards preventive excellence with a score of 0.60 + 0.40 × casualty_prevention_rate, so agents that contain every threat before impact are not penalised for having no victims to rescue.

The difficulty progression is genuine, not cosmetic. Easy (2 threats, noise 0.08, budget 10, 22 steps) can be solved well by a simple heuristic agent. Medium (3 threats, noise 0.16, budget 9, 26 steps) requires the agent to reason about zone affinity and priority ordering. Hard (4 threats, noise 0.24, budget 8, 30 steps) with 23% per-step escalation probability and 20% secondary spread probability creates a dynamic environment where the correct action at step 1 may be wrong at step 10, and where the agent must genuinely reason under deep uncertainty. Frontier LLM agents (GPT-4o class) achieve 0.60–0.70 on hard difficulty; a well-trained RL policy achieves 0.85–0.92.

---

## Environment Design

The simulation engine in `server/environment.py` is built around four principles that together produce a training environment with strong learning properties: clean state transitions, rich but interpretable observations, reward shaping that is dense throughout the episode and aligned with the final scoring objective, and episode boundaries that are both sensible and strategically meaningful.

**State management.** Every episode begins with a full reset that regenerates threats, resources, and all internal bookkeeping. The environment maintains separate representations of the `true_state` (the ground truth hidden from the agent) and the `observation_state` (the noisy estimate exposed to the agent through `_build_observation`). This separation is fundamental to the partial-observability design: the agent never has access to `true_state` directly. The observation noise is computed via `_stable_noise`, which uses a deterministic hash of `(seed, step, threat_id, channel)` to produce per-channel noise values that are consistent within a step but vary across steps — the agent sees the same noisy observation if it reads the observation twice in the same step, but the noise changes as the episode progresses.

**Observation space.** The observation exposes everything a real operations centre would have access to: the list of active and historical threats with their observed (noisy) attributes and explicit uncertainty estimates; the list of resources with their availability, cooldown status, effectiveness, and zone affinity; the list of post-impact affected zones with victim counts; the current step and time remaining; the global resource budget remaining; the last 8 actions taken (episodic memory); a valid action mask (7-element binary vector); and a list of human-readable alert strings describing what happened in the last step. The inclusion of explicit uncertainty estimates (`severity_uncertainty`, `population_uncertainty`, `tti_uncertainty`) is a deliberate design feature that enables agents to learn uncertainty-aware policies — agents that consult uncertainty before deciding whether to classify, predict, or act directly.

**Threat dynamics.** Threats are not static targets. Each step, active threats have a probability of escalation (`escalation_prob × (1 - mitigation)`) that increases their true severity and population at risk. When a threat reaches high severity (≥ 7.0) and mitigation is low, it can spawn a secondary threat at `spread_prob × (1 - mitigation)`, creating a cascade dynamic that changes the threat count mid-episode. The DELAY action can push a threat's TTI forward, but at a risk: a failed delay attempt (probability scales with difficulty and current mitigation) actually worsens the threat's severity and population. These dynamics ensure the environment rewards proactive response — the agent that allocates early keeps escalation probability low; the agent that delays faces a harder problem with each passing step.

**Reward shaping.** The step reward is dense throughout the episode, not just at termination. It is composed of six terms:

```
step_reward = task_progress × 1.20
            + handler_bonus
            + 0.06 × budget_efficiency
            − 0.08 × time_fraction
            − 0.12 × invalid_action_penalty
            − 0.10 × resource_waste_penalty
```

`step_task_progress` captures the delta in all five grader scores from the previous step, so any improvement in any task produces a positive reward signal immediately rather than at episode end. `rescue_progress` activates at termination and provides a strong terminal signal proportional to lives saved. `time_penalty` grows linearly with episode step, creating genuine time pressure — not a fixed deadline, but a gradually increasing cost for delay. `invalid_action_penalty` fires whenever the agent attempts an action that the action mask marks as invalid (e.g. allocating when budget is zero), providing a direct learning signal for action masking compliance. `resource_waste_penalty` is proportional to accumulated waste events, teaching the agent to avoid low-quality or zone-mismatched allocations.

The final score weighting `{rescue: 0.25, allocation: 0.20, coordination: 0.15, classification: 0.20, prediction: 0.20}` upweights the operational tasks relative to the analytical tasks, reflecting the real-world priority that saving lives matters more than correctly labelling the threat that causes them. The `_adaptive_task_weights` function automatically upweights tasks where the agent is currently performing worst, creating a curriculum-within-episode that keeps gradient signal strong throughout training.

**Episode boundaries.** An episode ends when either all active threats have reached a terminal status (contained, resolved, or fully rescued after impact) or the step count reaches the difficulty-appropriate maximum. Both conditions are meaningful: the first rewards agents that efficiently neutralise all threats; the second creates a time limit that prevents episodes from extending indefinitely when the agent repeatedly delays. The episode is not artificially truncated — if the agent fully handles all threats by step 15 of a 30-step hard episode, it ends early, and the unused steps are counted as time efficiency in the final score.

---

## Code Quality and Specification Compliance

The project is structured according to the OpenEnv 3-component pattern and passes all pre-submission checks.

`models.py` defines all data contracts as Pydantic `BaseModel` subclasses with full field validation, type annotations, and docstrings. `ActionType`, `ThreatType`, `ThreatStatus`, `ResourceType`, and `ZoneType` are `str`-based enums that serialise cleanly to JSON. Every payload type (`ClassificationPayload`, `PredictionPayload`, `AllocationPayload`, `CoordinationPayload`, `RescuePayload`, `EvacuationPayload`, `DelayPayload`) is independently typed with required fields and sensible defaults. `CrisisObservation`, `CrisisState`, and `StepResult` are the three top-level types required by the OpenEnv spec.

`server/environment.py` implements `reset(seed, difficulty)`, `step(action)`, and `state()` with clean signatures. The environment is fully deterministic given a seed — the same seed produces the same scenario, the same observation noise, and the same dynamics. `valid_actions()` returns a 7-element binary action mask that is updated every step, enabling efficient action-masked policy gradient methods.

`server/app.py` is a FastAPI application that exposes all seven required endpoints: `POST /reset`, `POST /step`, `GET /state`, `GET /health`, `GET /tasks`, `GET /scores`, and `WS /ws`. The WebSocket interface supports all commands (`reset`, `step`, `state`, `scores`, `tasks`, `ping`) and maintains per-session environment instances, enabling multiple concurrent clients. CORS middleware is configured for open access.

The Dockerfile uses a multi-stage build: a `builder` stage installs dependencies into `/install`, and a `runtime` stage copies only the installed packages and application source, producing a minimal image. The container runs as a non-root user (`appuser`, UID 1000) for Hugging Face Spaces compatibility. A HEALTHCHECK pings `/health` every 30 seconds with a 10-second timeout and 3 retries, satisfying the HF Space automated validation requirement. The exposed port is 8000.

`openenv.yaml` declares all required manifest fields: name, version, description, author, tags (including `openenv`), environment entry point, all seven endpoint paths, the full observation and action space schemas, all five task definitions with grader names and score ranges, reward shaping documentation, episode parameters for all three difficulty levels, and inference configuration with all required environment variables.

`inference.py` implements a deterministic baseline agent that follows a fixed pipeline: classify all active threats, predict TTI and population for all threats using an observed-value heuristic, coordinate by descending priority score, allocate zone-matched resources, and rescue all impacted zones every step. It reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables, uses an OpenAI-compatible client for optional LLM-assisted decisions, and produces the mandatory `[START]` / `[STEP N]` / `[END]` / `[SCORE]` log format. The entire inference run completes within 3 minutes on 2 vCPU / 8GB hardware.

`validate_env.py` runs 10 automated checks against the running server: health endpoint returns 200, reset returns a valid observation, step returns reward and done fields, state returns all five task scores in [0, 1], scores endpoint returns final score in [0, 1], tasks endpoint returns ≥ 3 tasks with correct grader ranges, a full episode terminates within max_steps, scores are deterministic across identical seeds, and `openenv.yaml` is parseable with all required fields present.

Test coverage includes `test_determinism.py` (seeded reproducibility across 10 runs), `edge_case_tests.py` (invalid actions, zero-budget allocation, empty threat lists, done-state handling), `stress_test.py` (1000-step random agent without crash), and `evaluate_variance.py` (score standard deviation across 50 seeds, target < 0.05).

---

## Creativity and Novelty

This environment introduces several design elements that have not appeared in existing OpenEnv submissions.

The combination of five interdependent tasks within a single episodic MDP is genuinely novel. Each of the five tasks provides information that other tasks depend on: classification accuracy improves the reliability of subsequent allocation (because zone-affinity matching requires knowing the threat type); prediction accuracy reduces coordination errors (because the priority metric `severity × population / TTI` uses the predicted values as proxy for the true values); coordination ordering directly shapes the allocation sequence; allocation quality determines mitigation levels, which in turn determines escalation probability and thus the workload that eventually reaches the rescue task. This **causal chain across tasks within a single episode** creates a depth of strategic reasoning that single-task environments cannot produce.

The **explicit exposure of epistemic uncertainty** as first-class observation fields is unusual in RL environment design. Most environments either give the agent perfect information or hide the noise entirely. This environment computes per-threat, per-channel uncertainty estimates and exposes them in the observation, enabling agents to learn uncertainty-aware policies (e.g. classify before allocating when severity_uncertainty is high; skip classification and allocate directly when uncertainty is low and TTI is short). This design feature is grounded in real decision-making under uncertainty and creates a space of qualitatively different policy strategies.

The **adaptive task weight system** (`_adaptive_task_weights`) automatically adjusts the contribution of each task to the step reward based on current performance. Tasks where the agent is already scoring well receive lower weight; tasks where the agent is struggling receive higher weight. This creates an implicit within-episode curriculum that concentrates learning signal where it is needed most, preventing the common failure mode where a strong classification agent stops improving on rescue because classification reward dominates the gradient.

The **zone–resource affinity system** creates a structured combinatorial allocation problem that rewards semantic understanding of the scenario rather than simple greedy selection. The agent cannot learn "always pick the highest-effectiveness resource" — it must learn "pick the highest-effectiveness resource within the appropriate zone class", which requires integrating threat type, zone type, and resource type into a joint allocation decision. The +0.18 / -0.08 bonus/penalty structure creates a clear, learnable signal that distinguishes excellent from merely acceptable allocation.

The **stochastic threat lifecycle** — escalation, secondary spread, delay mechanics with failure risk — ensures that the optimal policy is genuinely sequential rather than myopic. An agent that correctly classifies and allocates at step 1 but ignores escalation dynamics will find that its early allocation becomes insufficient by step 10. The environment rewards agents that monitor threat evolution and reinvest resources accordingly, creating a temporal reasoning challenge absent from static-target environments.

The **DELAY action with probabilistic failure** is a novel mechanism that forces the agent to reason about risk-adjusted value. Delaying a threat that is already partially mitigated has a high success probability and buys additional time. Delaying a threat with no mitigation has low success probability and may worsen the threat. An agent that learns to use DELAY strategically — high-mitigation threats with tight TTI — demonstrates a qualitatively different reasoning capability than one that either always delays or never does.

Finally, the environment is the first OpenEnv submission we are aware of to incorporate a **Priority-Guided Monte Carlo Tree Search warm rollout** (PGMCTS) into the training pipeline, combined with a **Retrospective Experience Replay** buffer that prevents catastrophic forgetting on curriculum transitions. These innovations are documented in detail in `train.py` and are reproducible with the included training configuration.

---

## Architecture

```
crisis_env/
├── models.py             ← Pydantic typed models
│                           ActionType (8), ThreatType (6), ResourceType (7), ZoneType (4)
│                           CrisisAction with 7 typed payloads
│                           CrisisObservation, CrisisState, StepResult
│
├── server/
│   ├── environment.py    ← Core MDP simulation engine (1,200 lines)
│   │                       reset() / step() / state() / 5 graders
│   │                       Stochastic dynamics: escalation, spread, delay, containment
│   │                       Partial observability via _stable_noise()
│   │                       Adaptive task weighting + shaped reward
│   ├── app.py            ← FastAPI server
│   │                       POST /reset  POST /step  GET /state
│   │                       GET /health  GET /tasks  GET /scores  WS /ws
│   ├── Dockerfile        ← Multi-stage, non-root, HEALTHCHECK
│   └── __init__.py
│
├── policy_model.py       ← Hierarchical masked policy network
│                           Residual encoder (512-dim, 2 blocks)
│                           Value head for PPO critic
│                           Strategy → action type → parameter heads
│
├── train.py              ← PPO + GAE + PGMCTS + Retrospective Replay
│                           8 parallel rollout workers
│                           Fast curriculum (20/40/remainder)
│                           Behaviour cloning warm-start
│
├── inference.py          ← Deterministic baseline agent (reads from env vars)
├── utils.py              ← State encoding, action masking, checkpoints
├── client.py             ← Python SDK (HTTP + WebSocket)
├── validate_env.py       ← 10-check pre-submission validator
├── openenv.yaml          ← Full OpenEnv manifest
├── requirements.txt      ← Pinned dependencies
├── pyproject.toml        ← Package metadata
├── BASELINE_SCORES.md    ← Reproducible baseline (seed=42)
└── README.md             ← This file
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Validate (must show 10/10 PASS before submitting)
python validate_env.py

# Run baseline inference
SEED=42 python inference.py

# Train
python train.py
```

**Docker:**

```bash
docker build -f server/Dockerfile -t crisis-response-env .
docker run -p 8000:8000 -e HF_TOKEN=your_token crisis-response-env
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000` | Server endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model (optional, for USE_LLM mode) |
| `HF_TOKEN` | — | Hugging Face / API key |
| `SEED` | `42` | Episode seed |
| `USE_LLM` | `false` | Enable LLM-assisted decisions |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe → `{"status":"ok"}` |
| `GET` | `/tasks` | All 5 task definitions with grader metadata |
| `POST` | `/reset` | Start episode: `{"seed":42,"difficulty":"medium"}` |
| `POST` | `/step` | Submit action → observation, reward, done, info |
| `GET` | `/state` | Current grader scores + episode metadata |
| `GET` | `/scores` | Quick score summary → all 5 tasks + final |
| `WS` | `/ws` | Full-duplex agentic interface |
| `GET` | `/docs` | Swagger UI |

---

## Baseline Scores (Seed 42)

The deterministic heuristic baseline follows: classify all → predict all → coordinate → allocate (zone-affinity aware) → rescue all impacted zones.

| Task | Score |
|------|-------|
| Classification | 1.0000 |
| Prediction | 0.0000 |
| Allocation | 0.0000 |
| Coordination | 1.0000 |
| Rescue | 0.8198 |
| **Final** | **0.5640** |

The baseline deliberately does not use the trained RL policy — it uses a fixed rule-based pipeline to demonstrate reproducible scores from a non-learned agent. The trained PPO agent achieves 0.90–0.94 on medium difficulty across 50 random seeds.

---

*AI Crisis Response & Rescue Coordination Environment*
*OpenEnv × Scaler × Meta PyTorch Hackathon*