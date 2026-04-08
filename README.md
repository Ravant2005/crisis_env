<div align="center">

# 🚨 Crisis-OpenEnv: AI Crisis Response & Rescue Coordination
### *A high-fidelity, real-world simulation for autonomous emergency coordination and life-saving optimization.*

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Tasks](https://img.shields.io/badge/Graded%20Tasks-5-orange)](./openenv.yaml)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

**Crisis-OpenEnv** models the high-stakes "Fog of War" experienced by Emergency Operations Centers (EOCs) during regional disasters. It is designed to evaluate whether AI agents can move beyond simple games and master the complex, interdependent pipeline of real-world crisis management: **Classification → Prediction → Allocation → Coordination → Rescue.**

</div>

## 📖 Environment Description & Motivation
In large-scale regional disasters (wildfires, floods, industrial accidents), Emergency Operations Centers (EOCs) are overwhelmed by simultaneous, evolving threats. Human operators face extreme cognitive load, leading to critical errors in triage and resource distribution. 

**Motivation:** This environment provides a reproducible benchmark for training AI agents to assist or automate the "initial response" phase of a crisis. It bridges the gap between toy RL environments and real-world operational challenges by introducing **Partial Observability**, **Stochastic Escalation**, and **Tight Resource Constraints**.

---

## 🛰️ Observation Space Definition
The observation space is a structured `CrisisObservation` (Pydantic model) that provides a comprehensive view of the disaster landscape:

| Field | Type | Description |
|---|---|---|
| `threats` | `List[ThreatInfo]` | Contains `threat_id`, `threat_type`, `status` (active/impacted/resolved), and **Noisy** readings for `severity`, `time_to_impact`, and `population_at_risk`. |
| `resources` | `List[ResourceInfo]` | Specialized units available for deployment (e.g., `fire_brigade`, `hazmat_team`). Includes `effectiveness` and `is_available` status. |
| `affected_zones` | `List[AffectedZoneInfo]` | Zones that have been impacted and require active rescue. Tracks `total_victims` and `rescued` counts. |
| `resource_budget_remaining` | `int` | The global budget for the episode (e.g., 8 units total). |
| `time_remaining` | `int` | Steps left until the episode terminates (max 30). |

---

## 🕹️ Action Space Definition
The agent must submit one `CrisisAction` per step. The environment enforces a logical pipeline:

1.  **`classify`**: Identify a threat's true type and severity based on noisy sensor data.
2.  **`predict`**: Forecast the time-to-impact (TTI) and total population at risk.
3.  **`allocate`**: Assign a specific resource unit to an active threat to reduce its impact.
4.  **`coordinate`**: Set a global priority ordering across all currently active threats.
5.  **`rescue`**: Deploy units to impacted zones to save lives (the final goal).
6.  **`skip`**: Pass the turn to allow time to advance (useful if waiting for better data).

---

## 🎯 Task Descriptions & Expected Difficulty

Crisis-OpenEnv features 5 interdependent tasks that an agent must master:

### 1. Threat Classification (Easy)
*   **Objective:** Correctly identify the type and severity of each active threat.
*   **Difficulty:** Low. Requires basic interpretation of noisy sensor inputs.
*   **Grader:** `0.70 * type_match + 0.30 * severity_accuracy`.

### 2. Impact Prediction (Medium)
*   **Objective:** Predict how much time remains before impact and how many lives are at risk.
*   **Difficulty:** Medium. Requires filtering temporal noise and estimating population density.
*   **Grader:** `1.0 - (0.5 * tti_error + 0.5 * pop_error)`.

### 3. Resource Allocation (Medium)
*   **Objective:** Match specialized assets (e.g., Coast Guard for floods) to the right threat.
*   **Difficulty:** Medium. Involves budget management and zone-affinity optimization.
*   **Grader:** `effectiveness * zone_affinity - waste_penalty`.

### 4. Multi-Threat Coordination (Hard)
*   **Objective:** Rank all threats by urgency to ensure critical incidents are handled first.
*   **Difficulty:** High. Requires global situational awareness and ranking correlation.
*   **Grader:** `SpearmanRankCorrelation(agent_order, ideal_order)`.

### 5. Rescue Optimization (Hard)
*   **Objective:** Deploy units to impacted zones to maximize lives saved.
*   **Difficulty:** Extreme. Agents must balance speed, budget, and victim counts.
*   **Grader:** `(0.85 * saved_ratio + 0.15 * speed_score) * urgency`.

---

## 📊 Baseline Scores
The following scores were achieved using the provided `inference.py` baseline agent:

| Task | Score | Result |
|---|---|---|
| **Classification** | 0.993 | ✅ Excellent |
| **Prediction** | 0.930 | ✅ Strong |
| **Allocation** | 0.900 | ✅ Optimized |
| **Coordination** | 1.000 | ✅ Perfect |
| **Rescue** | 0.428 | ⚠️ Weak (Area for Improvement) |
| **TOTAL FINAL SCORE** | **0.822** | **Target: 0.87 - 0.91** |

---

## 🚀 Setup & Usage Instructions

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone <your-repo-url>
cd crisis_env
pip install -r requirements.txt
```

### 2. Starting the Server
The environment runs as a FastAPI server. Start it using `uvicorn`:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Running Validation
Ensure your environment is compliant with the OpenEnv spec:
```bash
python validate_env.py
```
*Note: You should see "26/26 checks passed" before submitting.*

### 4. Running Baseline Inference
Run the baseline agent against the local server:
```bash
export API_BASE_URL="http://localhost:8000"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

### 5. Docker Deployment (Local or HF Spaces)
To build and run the containerized environment:
```bash
docker build -t crisis-env .
docker run -p 8000:8000 crisis-env
```

---

## ✅ Evaluation Criteria Mapping
*   **Real-world utility (30%):** Models genuine FEMA/EOC triage workflows.
*   **Task & Grader Quality (25%):** 5 tasks with deterministic programmatic graders.
*   **Environment Design (20%):** Clean state management and realistic stochastic dynamics.
*   **Code Quality (15%):** Fully typed models and 100% automated validation pass.
*   **Creativity (10%):** Novel "Fog of War" mechanics and pipeline-based reward shaping.

---
*Developed for the RL OpenEnv Competition. Ready for Phase 1-3 Evaluation.*
