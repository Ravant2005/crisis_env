---
title: CrisisAI OpenEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# 🚨 CrisisAI — AI Crisis Response & Rescue Coordination

> **OpenEnv RL Environment** | Multi-threat national emergency management with partial observability, stochastic dynamics, and five interlinked decision tasks grounded in real operational doctrine.

---

## 🌍 Motivation & Real-World Utility

Modern conflicts and disasters demand simultaneous, time-critical decisions across threat identification, casualty projection, resource deployment, inter-agency coordination, and survivor rescue — all under uncertainty and resource constraints. No existing RL benchmark captures this full pipeline in a single environment.

**CrisisAI** models a concrete, high-stakes scenario: multi-region crisis simulation —

| Threat | Vector | Location | Unmitigated Casualties |
|--------|--------|----------|----------------------|
| ✈️ Airstrike | Fighter jet | Military Zone Alpha | ~50 |
| 🚢 Naval Force | Ship attack | Maritime Sector 7 | ~300 |
| 🎯 Drone Strike | UAV swarm | Urban District / Business Zone | ~1,500 |

An agent must classify each threat, predict its impact, allocate the right defense resources, coordinate response priority across all three simultaneously, and rescue survivors if attacks land — under a shared resource budget and ticking TTI (time-to-impact) clock.

**Who benefits from this environment?**
- RL researchers studying multi-task, partially-observable sequential decision-making
- Defense and emergency management labs evaluating AI decision-support systems
- AI safety researchers studying goal prioritization under competing constraints and scarcity

---

## 🏗️ Environment Overview

```
CrisisAI OpenEnv  (3 registered tasks, 5 internal sub-tasks)
├── task_easy   → T1: Threat Classification
├── task_medium → T2: Impact Prediction
└── task_hard   → Full Crisis Response (T1 + T2 + T3 + T4 + T5 composite)
                      ├── T3: Resource Allocation
                      ├── T4: Multi-Threat Coordination
                      └── T5: Rescue Operations
```

**Key design properties:**

- **Partial observability** — all threat attributes carry calibrated Gaussian noise (±8% easy → ±24% hard); explicit per-field uncertainty values exposed to the agent
- **Stochastic dynamics** — threats escalate severity, spread to secondary zones, and can be contained each step based on mitigation level
- **Resource scarcity** — a shared integer budget (8–10 units by difficulty) must be rationed across allocation AND rescue; hoarding is penalised
- **Time pressure** — 22–30 steps per episode; late actions earn less reward via step-decay
- **Action masking** — invalid actions suppressed every step via `valid_actions.action_mask`; skip is masked whenever any productive action exists
- **Curriculum support** — three difficulty tiers with increasing noise, more threats, higher escalation/spread probability, and tighter budgets

---

## 🎯 Five Task Descriptions

All five sub-tasks are active in every episode. The three registered tasks expose them at increasing difficulty.

### T1 — Threat Classification (`classify`)
**What:** Identify each threat's type (airstrike, ship\_attack, drone\_threat, explosion, flood, fire) and estimate its severity on a 0–10 scale.
**Why it's hard:** Observations are noisy; severity drifts as threats escalate each step. A wrong type prediction scores 0; a domain match gives partial credit.
**Scored on:** Exact type match (0.55) + domain match partial (0.25) + severity Gaussian accuracy (0.20).
**Expected score:** ~0.70 (easy difficulty)

---

### T2 — Impact Prediction (`predict`)
**What:** For each active threat, predict time-to-impact (TTI in steps) and total population at risk.
**Why it's hard:** Both values are noisy and change as dynamics advance. The agent must predict through uncertainty, not just echo the observation.
**Scored on:** Normalised TTI error via Gaussian (σ=0.15) + log-ratio population error (σ=0.50).
**Expected score:** ~0.55 (medium difficulty)

---

### T3 — Resource Allocation (`allocate`)
**What:** Assign one of 8 available resource units (military\_unit, coast\_guard, swat\_team, fire\_brigade, medical\_team, rescue\_drone, evacuation\_bus) to each threat to intercept or mitigate it before impact.
**Why it's hard:** Zone–resource affinity matters (+0.18 mitigation bonus for matched pairs); budget is shared with rescue, so over-spending on allocation blocks survivor recovery. Mismatched resources waste budget.
**Scored on:** Mean resource effectiveness (0.45) + zone affinity match rate (0.30) + budget efficiency (0.15) − waste penalty.
**Expected score:** ~0.50 (hard difficulty)

---

### T4 — Multi-Threat Coordination (`coordinate`)
**What:** Submit a global priority ordering of all active threats from highest to lowest risk. The ordering can be re-submitted as the situation evolves (every ~5 steps).
**Why it's hard:** True priority = (severity × population) / TTI, which changes as threats escalate and spread. Stale ordering is penalised. Coordination score decays per step if not refreshed, so one early `coordinate` is not sufficient.
**Scored on:** Weighted rank-correlation vs. true priority order; penalty for mis-identifying the top-priority threat.
**Expected score:** ~0.55 (hard difficulty)

---

### T5 — Rescue Operations (`rescue`)
**What:** After a threat impacts and creates an affected zone, deploy rescue units to save survivors. Each unit saves approximately 14 × zone\_multiplier × effectiveness victims per step.
**Why it's hard:** Budget shared with T3 creates a real trade-off. Speed matters — earlier rescue earns more. Deploying too many units to one zone wastes budget needed elsewhere.
**Scored on:** Saved-victim ratio (0.70) + response speed (0.15) + resource efficiency per unit (0.15); progressive bonuses at 20% and 50% rescue rates.
**Expected score:** ~0.40 (hard difficulty)

---

## 📊 Task Registry

| Task ID | Name | Difficulty | Active Threats | Episode Steps | Expected Score |
|---------|------|------------|----------------|---------------|---------------|
| `task_easy` | Threat Classification | Easy | 2 | 22 | ~0.70 |
| `task_medium` | Impact Prediction | Medium | 3 | 26 | ~0.55 |
| `task_hard` | Full Crisis Response | Hard | 4 | 30 | ~0.35 |

---

## 🔍 Observation Space

Every `reset()` and `step()` returns a `CrisisObservation` serialised as a flat JSON dict.

### Global Fields

| Field | Type | Description |
|-------|------|-------------|
| `threats` | `List[ThreatInfo]` | Up to 6 threats with noisy attributes |
| `resources` | `List[ResourceInfo]` | 8 deployable units (type, zone, effectiveness, cooldown) |
| `affected_zones` | `List[AffectedZoneInfo]` | Post-impact zones awaiting rescue |
| `time_remaining` | `int` | Steps left in episode |
| `resource_budget_remaining` | `int` | Shared budget for allocation + rescue |
| `valid_actions` | `dict` | Action mask + valid IDs per action type |
| `alerts` | `List[str]` | Timestamped event log (escalation, impact, containment) |
| `recent_actions` | `List[str]` | Last 5 action types taken (action memory) |

### Per-Threat Observable Fields (with calibrated noise)
`threat_id`, `threat_type`, `status` [active/impacted/contained/resolved], `severity` [0–10], `population_at_risk`, `time_to_impact`, `zone` [military/maritime/urban/rural], `severity_uncertainty`, `population_uncertainty`, `tti_uncertainty`, `priority_score`, `risk_level` [low/medium/high], `recommended_action_hint`

### State Vector (for neural agents)
Flat `float32` array of dimension **453**: 14 global features + 6 threat slots × 29 + 8 resource slots × 15 + 6 zone slots × 11 + 4-step action history + action mask + units mask + strategy prior.

---

## ⚡ Action Space

One `CrisisAction` submitted per step. Seven action types:

| Action | Required Payload | Purpose |
|--------|-----------------|---------|
| `classify` | `threat_id`, `predicted_type`, `predicted_severity` | Identify and assess a threat |
| `predict` | `threat_id`, `predicted_tti`, `predicted_pop` | Forecast time-to-impact and population at risk |
| `allocate` | `threat_id`, `resource_id` | Assign a defense/rescue resource to intercept a threat |
| `coordinate` | `priority_order: List[int]` | Set global response priority across all active threats |
| `rescue` | `zone_id`, `rescue_units_to_send` | Deploy units to save survivors in impacted zones |
| `delay` | `threat_id`, `delay_steps` | Attempt stochastic TTI extension (can backfire and escalate) |
| `skip` | — | No-op; **only valid when no other action is possible** |

**Action masking** is built into every observation. `valid_actions.action_mask` is a binary list aligned to `[classify, predict, allocate, coordinate, rescue, skip, delay]`. Skip is masked out whenever any productive action is available.

---

## 🏆 Scoring & Reward Design

### Task Weights (composite final score)

| Sub-task | Weight |
|----------|--------|
| T1 · Threat Classification | 20% |
| T2 · Impact Prediction | 20% |
| T3 · Resource Allocation | 20% |
| T4 · Multi-Threat Coordination | 15% |
| T5 · Rescue Operations | **25%** |

### Grader Guarantees
- Returns scores in `(0.001, 0.999)` — never exactly 0 or 1, preserving gradient signal throughout training
- **Deterministic** — same seed → identical scores verified by `test_determinism.py` (3 independent runs)
- **Dual-mode** — accepts a live `env` object OR a flat `actions` list for offline platform validation
- **Varied** — grader smoke test confirms scores differ meaningfully across action quality levels

### Step Reward Signal (dense, not sparse)
```
reward = 3.0 × Σ(weight_i × Δtask_score_i) + action_handler_bonus − time_cost
```
Per-task shaped components: Gaussian-based classification accuracy, log-ratio prediction error, zone-affinity allocation quality, rank-correlation coordination score, victim-ratio rescue efficiency. Skip returns −0.12 (strong penalty to break no-op loops). Terminal bonus = 0.10 × final\_score.

---

## 🛠️ Setup & Usage

### Prerequisites
```
Python 3.10+  |  Docker  |  pip
```

### Install
```bash
cd crisis_env
pip install -r requirements.txt
```

### Run Server
```bash
# Local (port 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860

# OR via Docker
docker build -t crisis-env .
docker run -p 7860:7860 crisis-env
```

### Validate (OpenEnv Compliance — 18 checks)
```bash
python validate_env.py
# Expected: 18/18 PASS — STATUS: READY TO SUBMIT
```

### Run Inference (LLM Agent)
```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
# Runs task_easy → task_medium → task_hard
# Prints [START] / [STEP]* / [END] logs per OpenEnv spec
```

### Train RL Agent (PPO + GAE + PGMCTS)
```bash
python train.py --episodes 120 --n-workers 4 --device auto
```

### Determinism Test
```bash
python test_determinism.py   # seed=42, 3 runs → scores must match to <0.0001
```

### Stress Test (20 runs)
```bash
python stress_test.py        # PASS: mean > 0.80, std_dev < 0.10
```

### Python Client SDK
```python
from client import CrisisEnvClient

client = CrisisEnvClient("http://localhost:7860")
obs    = client.reset(seed=42)
client.classify(threat_id=1, predicted_type="airstrike", predicted_severity=7.5)
client.predict(threat_id=1, predicted_tti=5, predicted_pop=400)
client.coordinate(priority_order=[3, 2, 1])          # drone > ship > airstrike
client.allocate(threat_id=3, resource_id=4)           # fire_brigade → drone zone
client.rescue(zone_id=3, units=5)
state = client.state()   # {"classification_score": ..., "final_score": ...}
```

### REST API
```
GET  /health   → {"status": "ok"}
GET  /tasks    → 3 task definitions with grader_range
POST /reset    → {"task_id": "task_hard", "seed": 42}  → CrisisObservation
POST /step     → {"action": {...}}  → {observation, reward, done, info}
GET  /state    → all 5 task scores + full episode metadata
GET  /scores   → fast score-only summary
```

Swagger UI available at `http://localhost:7860/docs`.

---

## 📈 Baseline Scores

Measured with the rule-based heuristic baseline (`utils.run_local_baseline_episode`) — 20 seeds per difficulty:

| Sub-task | Easy | Medium | Hard |
|----------|------|--------|------|
| T1 · Classification | ~0.72 | ~0.68 | ~0.61 |
| T2 · Prediction | ~0.58 | ~0.52 | ~0.44 |
| T3 · Allocation | ~0.61 | ~0.55 | ~0.48 |
| T4 · Coordination | ~0.60 | ~0.54 | ~0.46 |
| T5 · Rescue | ~0.45 | ~0.38 | ~0.30 |
| **Final (composite)** | **~0.58** | **~0.51** | **~0.42** |

The trained PPO + PGMCTS agent targets **≥0.88** composite on hard. An untrained frontier LLM (Qwen2.5-72B) scores approximately **0.35–0.40** on `task_hard`, confirming that the hard task genuinely challenges state-of-the-art models.

---

## 🗂️ Project Structure

```
crisis_env/
├── server/
│   ├── app.py                      # FastAPI server — OpenEnv endpoints
│   ├── environment.py              # Core MDP simulation engine (5 graders, dynamics)
│   └── crisis_env_environment.py   # OpenEnv interface wrapper
├── models.py          # Pydantic v2 typed models: Action, Observation, State
├── grader.py          # TASK_REGISTRY + 5 deterministic grader functions
├── rewards.py         # Per-task shaped reward functions + rich console dashboard
├── policy_model.py    # Hierarchical masked PPO policy network (256-dim)
├── train.py           # PPO + GAE + PGMCTS + curriculum learning training loop
├── inference.py       # LLM agent inference — produces OpenEnv platform log format
├── utils.py           # State vectorisation (453-dim), action codec, baseline agent
├── client.py          # Synchronous Python client SDK
├── validate_env.py    # 18-check pre-submission OpenEnv compliance validator
├── test_determinism.py # Seed reproducibility test (3 runs, <0.0001 tolerance)
├── stress_test.py     # 20-run performance and stability statistics
├── openenv.yaml       # OpenEnv manifest
├── Dockerfile         # Container build for HF Spaces deployment
└── requirements.txt
```

---

## ✨ Design Highlights

**Novel domain:** Multi-domain simultaneous crisis response (air + maritime + urban civilian) with a five-stage interlinked decision pipeline — not represented in existing OpenEnv submissions.

**Clever reward mechanics:** Each sub-task has a domain-specific shaped reward — Gaussian classification accuracy, log-ratio population prediction, zone-affinity resource matching, weighted rank-correlation for coordination, progressive rescue rate bonuses. All produce dense, varying signal rather than sparse end-of-episode rewards.

**Engaging environment dynamics:** Secondary threat spawning (high-severity threats cascade into adjacent zones), stochastic delay actions (can backfire and escalate), resource cooldown cycles, and a shared budget that creates a genuine allocation vs. rescue trade-off — forcing the agent to plan multiple steps ahead under real constraints.

**Research-grade infrastructure:** Hierarchical masked policy network, PPO + GAE + PGMCTS planner, retrospective experience replay, behavior cloning warm-start, curriculum learning (easy → medium → hard), and a full test suite — ready to use for agent research out of the box.

---

*Built for the Meta × Hugging Face × Scalar OpenEnv Hackathon.*