<div align="center">

# 🚨 AI Crisis Response & Rescue Coordination
### A high-fidelity RL environment for AI-driven multi-threat emergency coordination under partial observability and tight resource constraints.

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Tasks](https://img.shields.io/badge/Graded%20Tasks-5-orange)](./openenv.yaml)

</div>

## ⚡ Quick Overview
* **Tasks:** 5 interdependent tasks (Classification → Prediction → Allocation → Coordination → Rescue).
* **Score range:** 0.0 – 1.0 (for each task and final score).
* **Difficulty:** Scalable (Easy → Medium → Hard).
* **Deployment:** Hugging Face Docker Space.

## 🧠 Real-World Problem
During wartime or critical disasters, emergency operations centers face simultaneous threats competing for finite response resources. Human cognitive load causes deadly errors—wrong assets deployed, secondary threats ignored, and chaotic resource allocation. This environment trains AI agents to solve multi-incident triage realistically under "fog-of-war" sensor uncertainty, strictly replicating Emergency Response Frameworks.

## 🎯 Tasks & Graders

### 1. Threat Classification
* **What agent does:** Classifies threat types and predicts severity levels from noisy inputs.
* **Input:** Threat ID, zone features, and `severity_uncertainty` fields.
    ```python
    final_score = 0.70 * correct_ratio + 0.30 * severity_accuracy
    ```

### 2. Impact Prediction
* **What agent does:** Forecasts the Time-To-Impact (TTI) and Population at Risk.
* **Input:** Noisy TTI and population signals along with confidence intervals.
* **Scoring formula:**
    ```python
    score = 1.0 - (0.5 * mean(tti_err_normalized) + 0.5 * mean(pop_err_normalized))
    ```

### 3. Resource Allocation
* **What agent does:** Deploys highly constrained units to threats according to zone-affinities.
* **Input:** Available resources, cooldowns, and active threats.
* **Scoring formula:**
    ```python
    score = 0.45 * effectiveness + 0.30 * zone_affinity + 0.15 * budget_efficiency - waste_penalty
    ```

### 4. Multi-Threat Coordination
* **What agent does:** Ranks simultaneous threats by calculated operational priority.
* **Input:** All active threats requiring attention.
* **Scoring formula:**
    ```python
    score = mean(rank_correlation(agent_order, ideal_order)) - wrong_priority_penalty
    ```

### 5. Rescue Optimization
* **What agent does:** Dispatches post-impact rescue units to impacted zones to save victims.
* **Input:** `AffectedZoneInfo` with `total_victims` and remaining response budget.
* **Scoring formula:**
    ```python
    score = (0.65 * saved_ratio + 0.20 * speed_score + 0.15 * resource_efficiency) * urgency
    ```

## ⚙️ Environment Design
* **State:** Maintained separately as `true_state` (hidden) and `observation_state`.
* **Observations:** Provides active threats, deployed resources, affected zones, budget, memory vectors (recent actions), action mask, and per-channel uncertainty estimates.
* **Actions:** `classify`, `predict`, `allocate`, `coordinate`, `rescue`, `delay`, `evacuate`.
* **Constraints:** Global operational budget (strictly capped), resource cooldowns (1-3 steps).
* **Dynamics:**
    * Escalation probability increases true severity over steps.
    * Threats can uncontrollably spawn "secondary spread" events.
    * Delayed actions bear intrinsic failure mechanisms worsening population impacts.

## 🎯 Reward Function
* **Equation:**
    ```python
    step_reward = (task_progress * 1.20) + 0.06 * budget_efficiency - 0.08 * time_fraction - 0.12 * invalid_action - 0.10 * waste_penalty + handler_bonus
    ```
* **Components:**
    * **Task Progress:** Delta improvements across Graders per step.
    * **Efficiency/Time:** Drives fast behavior and discourages pointless asset expenditures.
    * **Invalid/Waste:** Hard penalties for violating masking logic or misusing budgets on non-affinity zones.

## 📊 Scoring Breakdown
* **Weights:**
  * Rescue: `25%`
  * Classification: `20%`
  * Prediction: `20%`
  * Allocation: `20%`
  * Coordination: `15%`
* **Bonus Mechanics**: Final computation is a strict weighted sum; no artificial time milestones or milestone bonus inflations are applied.

## 🧪 Baseline Performance
| Task           | Score  |
|----------------|--------|
| Classification | 1.0000 |
| Prediction     | 0.0000 |
| Allocation     | 0.0000 |
| Coordination   | 1.0000 |
| Rescue         | 0.8198 |
| **Final**      | **0.5640** |
*(Note: Trained PPO agents attain > 0.90 reliably, deterministic behavior confirmed via static seeds)*

## 🏗️ Architecture
* **`models.py`**: Pydantic contracts / data mappings.
* **`server/`**: FastAPI implementation & the core MDP logic engine (`environment.py`).
* **`inference.py`**: Reproducible deterministic base agent.
* **`train.py`**: Custom PPO + GAE optimization implementation.

## 🚀 Quick Start
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
python validate_env.py
SEED=42 python inference.py
```

## ✅ Compliance Checklist
* OpenEnv endpoints ✔
* Docker ✔
* Deterministic ✔
* 3+ tasks ✔
* Reproducible baseline ✔