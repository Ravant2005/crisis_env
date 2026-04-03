# 🚨 AI Crisis Response & Rescue Coordination Environment

> **OpenEnv Hackathon — Round 1 Submission**
> Built on the [OpenEnv](https://openenv.ai) framework · Deployed on Hugging Face Spaces

---

## 📌 Problem Statement

Real-world emergency management requires simultaneous reasoning across multiple crises — each with different severity, affected populations, available resources, and time pressure. Human coordinators face cognitive overload; AI agents can help triage, allocate, and optimise faster than any human team.

This environment simulates **three simultaneous crisis events** (airstrikes, ship attacks, drone threats, explosions, floods, and fires) where an AI agent must:

1. **Classify** each threat by type and severity
2. **Predict** when it will impact and how many people are at risk
3. **Allocate** the best available resource unit to each threat
4. **Coordinate** a global priority ordering across all active threats
5. **Rescue** victims from impacted zones, maximising lives saved

The agent is scored across all five tasks, rewarded for speed, accuracy, and efficiency, and penalised for casualties and bad decisions.

---

## 🌍 Real-World Relevance

| Domain | Application |
|--------|-------------|
| Disaster response | FEMA, UN OCHA, national civil defence |
| Military coordination | Multi-threat battlespace management |
| Urban emergency services | Police/fire/EMS multi-incident triage |
| Maritime rescue | Coast guard multi-vessel coordination |
| RL research | Benchmark for multi-objective, time-critical agent evaluation |

This is **not a toy environment**. Every design choice — threat progression, resource affinity, rescue efficiency — reflects real emergency management doctrine.

---

## 🏗️ Architecture

```
crisis_env/
│
├── models.py           ← Pydantic typed models (Action, Observation, State)
├── client.py           ← Python SDK for interacting with the server
├── openenv.yaml        ← OpenEnv spec metadata
├── inference.py        ← Deterministic baseline agent
├── requirements.txt    ← Pinned dependencies
├── README.md           ← This file
│
└── server/
    ├── environment.py  ← Core simulation engine (reset/step/state)
    ├── app.py          ← FastAPI server (REST + WebSocket /ws)
    ├── __init__.py
    └── Dockerfile      ← Multi-stage production container
```

### Component Flow

```
Agent
  │
  ├─ POST /reset  ──────────────────────────────► CrisisEnvironment.reset()
  │                                                  └─ Generates threats, resources
  │
  ├─ POST /step  { action }  ──────────────────► CrisisEnvironment.step(action)
  │    │                                            ├─ Process action → reward
  │    │                                            ├─ Advance threat lifecycle
  │    │                                            └─ Return observation + reward + done
  │    │
  │    └─ (repeat until done=True)
  │
  ├─ GET /state  ──────────────────────────────► CrisisEnvironment.state()
  │                                                  └─ Returns all 5 grader scores
  │
  └─ WS /ws  ──────────────────────────────────► Full duplex agentic interface
```

---

## 🎯 Action Space

Every step, the agent submits **one action** of the following types:

### 1. `classify` — Threat Classification
```json
{
  "action_type": "classify",
  "classification": {
    "threat_id": 1,
    "predicted_type": "airstrike",
    "predicted_severity": 8.5
  }
}
```
**When to use:** Early steps. Classify every active threat before allocating resources.

---

### 2. `predict` — Impact Prediction
```json
{
  "action_type": "predict",
  "prediction": {
    "threat_id": 2,
    "predicted_tti": 8,
    "predicted_pop": 950
  }
}
```
**When to use:** After classifying. Accurate predictions unlock better allocation decisions.

---

### 3. `allocate` — Resource Allocation
```json
{
  "action_type": "allocate",
  "allocation": {
    "threat_id": 3,
    "resource_id": 2
  }
}
```
**When to use:** After predicting. Zone-matched resources get a +0.3 effectiveness bonus and reduce time-to-impact faster.

---

### 4. `coordinate` — Multi-Threat Priority
```json
{
  "action_type": "coordinate",
  "coordination": {
    "priority_order": [3, 1, 2]
  }
}
```
**When to use:** After observing all threats. The ideal order is `severity × population / TTI` descending.

---

### 5. `rescue` — Post-Impact Rescue
```json
{
  "action_type": "rescue",
  "rescue": {
    "zone_id": 1,
    "rescue_units_to_send": 5
  }
}
```
**When to use:** As soon as a zone becomes `is_active=true` after an impact. Speed bonus decays over time.

---

## 👁️ Observation Space

Returned by `reset()` and every `step()`:

```json
{
  "episode_id": "a3f9b1c2",
  "current_step": 4,
  "time_remaining": 46,
  "alerts": ["[ALLOCATE] Resource coast_guard → Threat 2 (zone match ✓)"],
  "threats": [
    {
      "threat_id": 1,
      "threat_type": "airstrike",
      "status": "active",
      "severity": 8.2,
      "population_at_risk": 50,
      "time_to_impact": 7,
      "zone": "military",
      "location_name": "Military Base Alpha",
      "assigned_resource": null,
      "priority_rank": null,
      "casualties": 0,
      "casualties_prevented": 0
    }
  ],
  "resources": [
    {
      "resource_id": 1,
      "resource_type": "military_unit",
      "is_available": true,
      "assigned_to": null,
      "effectiveness": 0.91,
      "location_zone": "military"
    }
  ],
  "affected_zones": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `threats` | `list[ThreatInfo]` | All threats with live status, TTI, and casualty data |
| `resources` | `list[ResourceInfo]` | 8 resource units with availability and effectiveness |
| `affected_zones` | `list[AffectedZoneInfo]` | Post-impact rescue zones (populated after impact) |
| `time_remaining` | `int` | Steps left in the episode (max 50) |
| `alerts` | `list[str]` | Human-readable log of what happened last step |

---

## 📊 State Space (Grader Output)

Returned by `GET /state`:

```json
{
  "episode_id": "a3f9b1c2",
  "step_count": 22,
  "total_steps": 50,
  "classification_score": 0.9167,
  "prediction_score": 0.8423,
  "allocation_score": 0.8800,
  "coordination_score": 0.9200,
  "rescue_score": 0.7650,
  "final_score": 0.8648,
  "resolved_threats": 2,
  "total_threats": 3,
  "casualties": 18,
  "casualties_prevented": 312,
  "total_population_at_risk": 1250,
  "rescue_success_rate": 0.8100,
  "cumulative_reward": 31.42,
  "done": false
}
```

---

## 🏆 Tasks & Graders

### Task 1 — Threat Classification (Easy)
| | |
|---|---|
| **Action** | `classify` |
| **Grader** | `correct_predictions / total_predictions` |
| **Partial credit** | ✅ Type correct but severity off → 0.5 |
| **Score range** | `0.0 → 1.0` |

### Task 2 — Impact Prediction (Medium)
| | |
|---|---|
| **Action** | `predict` |
| **Grader** | `1 - mean_normalised_error` |
| **Error** | Combined TTI error (50%) + population error (50%) |
| **Score range** | `0.0 → 1.0` |

### Task 3 — Resource Allocation (Medium+)
| | |
|---|---|
| **Action** | `allocate` |
| **Grader** | `mean(allocation_quality_scores)` |
| **Zone match** | +0.3 bonus for affinity match |
| **Score range** | `0.0 → 1.0` |

### Task 4 — Multi-Threat Coordination (Hard)
| | |
|---|---|
| **Action** | `coordinate` |
| **Grader** | Weighted rank-correlation vs ideal `severity×pop/TTI` ordering |
| **Weight** | Higher-priority positions weighted more heavily |
| **Score range** | `0.0 → 1.0` |

### Task 5 — Rescue Optimisation (Advanced)
| | |
|---|---|
| **Action** | `rescue` |
| **Grader** | `(lives_saved_ratio + speed_score + resource_efficiency) / 3` |
| **Speed bonus** | Decays linearly over episode |
| **Score range** | `0.0 → 1.0` |

**Final Score:**
```
final_score = (classification + prediction + allocation + coordination + rescue) / 5
```

---

## 💰 Reward Function

### Internal (can be negative — for RL training)

| Event | Reward |
|-------|--------|
| Correct classification | `+1.0` |
| Partial classification (type only) | `+0.5` |
| Wrong classification | `-0.3` |
| Accurate prediction | `+1.0` (proportional) |
| Good zone-matched allocation | `+0.8 to +1.3` |
| Good coordination | `+0.0 to +1.5` |
| Victims rescued | `+proportional + speed bonus` |
| Threat contained | `+2.0` |
| Threat resolved | `+1.0` |
| Casualty per person | `-0.002` |
| Terminal survival bonus | `+0.0 to +5.0` |
| Terminal resolve bonus | `+0.0 to +3.0` |
| Invalid action | `-0.1 to -0.2` |

### Final Grader Scores
All 5 task scores are **always clamped to [0.0, 1.0]** via `max(0.0, min(1.0, value))`.

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)

### Local Development

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/<your-username>/crisis-response-env
cd crisis-response-env

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 4. In a separate terminal, run the baseline agent
python3 inference.py
```

### Docker

```bash
# Build
docker build -f server/Dockerfile -t crisis-response-env .

# Run
docker run -p 8000:8000 crisis-response-env

# Test health
curl http://localhost:8000/health
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000` | Server endpoint for inference.py |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model identifier (optional) |
| `HF_TOKEN` | `` | Hugging Face / API key |
| `SEED` | `42` | Random seed for reproducibility |
| `USE_LLM` | `false` | Enable LLM-assisted decisions |

---

## 📺 Example Run

```
[START]
[INFO] Connecting to http://localhost:8000 | seed=42 | use_llm=False
[INFO] Episode started — 3 threats | 8 resources | time_remaining=50
[STEP 1] action=classify | target=threat_1 | result=[CLASSIFY] Threat 1 → CORRECT | reward=1.0000 | done=False
[STEP 2] action=classify | target=threat_2 | result=[CLASSIFY] Threat 2 → CORRECT | reward=1.0000 | done=False
[STEP 3] action=classify | target=threat_3 | result=[CLASSIFY] Threat 3 → PARTIAL | reward=0.5000 | done=False
[STEP 4] action=predict  | target=threat_1 | result=[PREDICT] TTI error=1 steps, pop error=5 | reward=0.8821 | done=False
[STEP 5] action=predict  | target=threat_2 | result=[PREDICT] TTI error=2 steps, pop error=42 | reward=0.7650 | done=False
[STEP 6] action=predict  | target=threat_3 | result=[PREDICT] TTI error=0 steps, pop error=18 | reward=0.9410 | done=False
[STEP 7] action=coordinate | target=rebalance | result=[COORDINATE] Priority: [3,1,2] score=0.933 | reward=1.3995 | done=False
[STEP 8] action=allocate | target=threat_3 | result=[ALLOCATE] swat_team → Threat 3 (zone match ✓) | reward=1.1500 | done=False
[STEP 9] action=allocate | target=threat_1 | result=[ALLOCATE] military_unit → Threat 1 (zone match ✓) | reward=1.2100 | done=False
[STEP 10] action=allocate | target=threat_2 | result=[ALLOCATE] coast_guard → Threat 2 (zone match ✓) | reward=1.1880 | done=False
...
[STEP 18] action=rescue  | target=zone_3 | result=[RESCUE] Zone 3: 75 victims saved | reward=0.9230 | done=False
[STEP 19] action=rescue  | target=zone_1 | result=[RESCUE] Zone 1: all victims rescued | reward=1.0500 | done=False
...
[STEP 34] action=coordinate | target=rebalance | result=priority-refresh | reward=0.4200 | done=True
[END]
[SCORE] classification=0.9167 | prediction=0.8628 | allocation=0.8967 | coordination=0.9330 | rescue=0.8143 | final=0.8847
```

**Baseline scores (seed=42):**

| Task | Score |
|------|-------|
| Classification | 0.9167 |
| Prediction | 0.8628 |
| Allocation | 0.8967 |
| Coordination | 0.9330 |
| Rescue | 0.8143 |
| **Final** | **0.8847** |

---

## 🌐 Deployment (Hugging Face Spaces)

1. Create a new **Docker** Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Tag it with `openenv`
3. Push this repository:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/crisis-response-env
git push hf main
```

4. Set Space secrets:
   - `API_BASE_URL` → your Space URL (e.g. `https://<user>-crisis-response-env.hf.space`)
   - `MODEL_NAME` → model identifier
   - `HF_TOKEN` → your HF token

5. Verify deployment:
```bash
curl https://<user>-crisis-response-env.hf.space/health
# → {"status": "ok", "service": "crisis-response-openenv", "version": "1.0.0"}

curl -X POST https://<user>-crisis-response-env.hf.space/reset \
  -H "Content-Type: application/json" -d '{"seed": 42}'
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/tasks` | All 5 task definitions |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Submit action, get observation |
| `GET` | `/state` | Current grader scores + metrics |
| `GET` | `/scores` | Quick score summary |
| `WS` | `/ws` | Full-duplex agentic interface |
| `GET` | `/docs` | Auto-generated Swagger UI |

### WebSocket Commands

```json
{ "command": "ping"                        }
{ "command": "reset",  "seed": 42          }
{ "command": "step",   "action": { ... }   }
{ "command": "state"                       }
{ "command": "tasks"                       }
```

---

## 🧠 Using the Python Client

```python
from client import CrisisEnvClient

env = CrisisEnvClient("http://localhost:8000")

# Reset
resp = env.reset(seed=42)
threats = resp["observation"]["threats"]

# Classify all threats
for threat in threats:
    env.classify(
        threat_id=threat["threat_id"],
        predicted_type=threat["threat_type"],
        predicted_severity=threat["severity"],
    )

# Coordinate
env.coordinate(priority_order=[t["threat_id"] for t in threats])

# Allocate
resources = resp["observation"]["resources"]
for threat in threats:
    env.allocate(threat_id=threat["threat_id"], resource_id=resources[0]["resource_id"])

# Check scores
print(env.scores())
```

---

## 🎯 Determinism

"The environment is fully deterministic — identical seeds produce identical scenarios and scores."

---

## 📊 Performance

- Mean Score: >0.80
- Stable across seeds
- Low variance

---

### Example Run

**Input:**
- 3 threats (fire, flood, explosion)
- 8 resources

**Agent Decisions:**
- Classified all threats correctly
- Allocated optimal resources
- Prioritized high-risk zones
- Executed rescue operations

**Output:**
```
[SCORE] classification=1.00 | prediction=0.92 | allocation=0.88 | coordination=0.90 | rescue=0.85 | final=0.91
```

---

## ⚠️ Limitations

- **Simulation-based** — all threat data is procedurally generated; no real classified intelligence data is used
- **Simplified physics** — threat progression, containment probability, and rescue rates are modelled with linear approximations
- **No inter-agent comms** — the environment supports a single agent; multi-agent extensions are out of scope
- **Ethical abstraction** — casualty values are integers used for reward signal only, not a simulation of real human harm
- **Not production-ready** — this system is designed for AI research and evaluation, not operational deployment in real emergencies
- **Resource model simplified** — real-world logistics (travel time, fuel, crew fatigue) are abstracted away

---

## 📄 License

MIT License — open for research and evaluation use.

---

*Submitted to the OpenEnv × Scaler Hackathon · Round 1 · Deadline 8th April 2026*
