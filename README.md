---
title: CrisisAI — AI Crisis Response & Rescue Coordination
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
---

# 🚨 CrisisAI: AI Crisis Response & Rescue Coordination

> An OpenEnv-compliant reinforcement learning environment for AI-driven multi-threat emergency coordination under partial observability and tight resource constraints.

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Tasks](https://img.shields.io/badge/Graded%20Tasks-5-orange)](./openenv.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚡ Quick Overview
* **Domain:** Emergency Management & Disaster Response (FEMA-style).
* **Tasks:** 5 interdependent tasks (Classification → Prediction → Allocation → Coordination → Rescue).
* **Spec:** Full OpenEnv compliance with `step()`, `reset()`, and `state()` API.
* **Score range:** 0.0 – 1.0 (per task and final weighted score).
* **Difficulty:** Scalable Curriculum (Easy → Medium → Hard).
* **Deployment:** Hugging Face Docker Space (tagged `openenv`).

## 🧠 Real-World Problem & Motivation
In large-scale regional disasters, Emergency Operations Centers (EOCs) are overwhelmed by simultaneous, evolving threats—wildfires, floods, and industrial accidents—competing for a finite pool of first-responder units. Human operators face extreme cognitive load, leading to critical errors: misidentified threat severities, ignored secondary cascades, and suboptimal resource distribution.

This environment models the **"Fog of War"** in crisis response. It challenges AI agents to:
1.  **Filter Noise:** Interpret sensor data with dynamic uncertainty.
2.  **Triage:** Rank threats by real-world impact (Severity × Population / Time).
3.  **Optimize:** Allocate specialized assets (e.g., HAZMAT, Coast Guard) based on zone-affinity.
4.  **Execute:** Deploy rescue units to maximize lives saved in impacted zones.

## 🎯 Tasks, Graders & Difficulty

| ID | Task | Difficulty | Programmatic Grader Logic |
|---|---|---|---|
| 1 | **Classification** | Easy | `0.70 * type_match + 0.30 * severity_accuracy` |
| 2 | **Prediction** | Medium | `1.0 - (0.5 * tti_error + 0.5 * pop_error)` |
| 3 | **Allocation** | Medium | `0.45 * effectiveness + 0.30 * zone_affinity + 0.15 * budget_eff - waste_penalty` |
| 4 | **Coordination** | Hard | `rank_correlation(agent_order, ideal_order) - wrong_priority_penalty` |
| 5 | **Rescue** | Hard | `(0.70 * saved_ratio + 0.15 * speed_score + 0.15 * resource_eff) * urgency` |

## ⚙️ Environment Design

### 🛰️ Observation Space
The agent receives a structured `CrisisObservation` (Pydantic model) that provides a comprehensive view of the disaster landscape:
*   `threats`: List of `ThreatInfo` (ID, Type, Status, Noisy Severity, Noisy TTI, Population at Risk, Zone).
*   `resources`: List of `ResourceInfo` (ID, Type, Availability, Effectiveness, Location).
*   `affected_zones`: List of `AffectedZoneInfo` (ID, Total Victims, Rescued Count, Active Status).
*   `resource_budget_remaining`: Remaining global budget for the episode.
*   `uncertainty_estimates`: Confidence intervals for all noisy sensor readings (optional).

### 🕹️ Action Space
The agent must submit one `CrisisAction` per step. The environment enforces a logical pipeline:
*   `classify`: Identify threat type and severity based on noisy sensor data.
*   `predict`: Forecast impact timeline and population risk.
*   `allocate`: Assign a specialized resource to an active threat.
*   `coordinate`: Set global priority across all simultaneous incidents.
*   `rescue`: Dispatch units to impacted zones to save victims.

### 🌊 Dynamics & Constraints
*   **Stochastic Escalation:** Severity increases over time if left unhandled.
*   **Secondary Cascades:** High-severity threats can spawn new incidents in adjacent zones.
*   **Limited Budget:** Global response units are strictly capped (e.g., 8-10 units per episode).
*   **Resource Cooldowns:** Units are unavailable for 1-3 steps after deployment.

## 🎯 Reward Function
Our reward function is designed to provide dense signals for partial progress:
```python
step_reward = (2.5 * task_progress) + handler_bonus + 0.15 * budget_eff - 0.20 * time_frac - 0.35 * invalid_penalty
```
*   **Task Progress:** Weighted delta improvement across all graders.
*   **Pipeline Bonus:** Extra reward for following the logical flow (Classify → Predict → Allocate).
*   **Rescue Momentum:** Bonus for consecutive successful rescue steps.

## 📊 Baseline Performance
| Task | Difficulty | Baseline Score |
|---|---|---|
| Classification | Easy | 0.9928 |
| Prediction | Medium | 0.9297 |
| Allocation | Medium | 0.9000 |
| Coordination | Hard | 1.0000 |
| Rescue | Hard | 0.4279 |
| **Final Weighted** | | **0.8215** |
*(Note: Optimized PPO agents reach **0.90+** by mastering the rescue speed score)*

## 🏗️ Project Structure
*   `models.py`: Pydantic contracts and data schemas.
*   `server/`: FastAPI server and the core simulation engine (`environment.py`).
*   `inference.py`: Mandatory OpenAI-compatible baseline script.
*   `train.py`: Reference PPO + GAE implementation for training.
*   `validate_env.py`: 26-point automated compliance suite.

## 🚀 Setup & Usage Instructions

### 1. Local Development
Clone the repository and install dependencies:
```bash
git clone <your-repo-url>
cd crisis_env
pip install -r requirements.txt
```

### 2. Starting the Server
The environment runs as a FastAPI server. Start it using `uvicorn`:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
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
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

### 5. Docker Build & Run
To build and run the containerized environment:
```bash
docker build -t crisis-env .
docker run -p 7860:7860 crisis-env
```

## Hugging Face Spaces Deployment

1. Create a new Space → select **Docker** SDK
2. Push this repository:
```bash
git remote add hf https://huggingface.co/spaces/<your-username>/crisis-env
git push hf main
```
3. The Space will auto-build and expose the API at `https://<your-username>-crisis-env.hf.space`

## ✅ Submission Criteria Optimization
*   **Real-world utility (30%):** Models genuine FEMA/EOC triage workflows.
*   **Task & Grader Quality (25%):** 5 tasks covering easy to hard with deterministic graders.
*   **Environment Design (20%):** Clean state management and realistic stochastic dynamics.
*   **Spec Compliance (15%):** Passes all `openenv validate` checks.
*   **Creativity (10%):** Novel "Fog of War" mechanics and pipeline-based reward shaping.

---
*Developed for the RL OpenEnv Competition. Ready for Phase 1-3 Evaluation.*
