---
title: "openenv"
emoji: "⚡"
colorFrom: "yellow"
colorTo: "green"
sdk: "docker"
app_port: 7860
short_description: "OpenEnv Smart Solar Grid Baseline System"
---

# ⚡ OpenEnv Smart Solar AI Grid

This is an **OpenEnv-compliant API Environment** designed to simulate and evaluate AI agent reasoning within a real-world smart grid energy management system.

## 🌍 Environment Description & Motivation
The **Smart Solar Env** tasks agents with a mission-critical real-world problem: intelligently distributing solar energy across a neighborhood grid, storing excess energy in batteries during peak sunlight, and mitigating rolling blackouts during heavy cloud coverage or nighttime. 

**Motivation:** Modern power grids are struggling to balance renewable energy volatility. This environment challenges frontier LLMs and RL agents to solve dynamic energy-routing optimization problems using real-world real-time weather data streams (integrated via Open-Meteo).

## 📊 Observation & Action Spaces

### Observation Space
The environment emits a typed state dictionary mapping exactly to `EnergyObservation`:
- `step` (int): Current hour/step of the simulation.
- `time_of_day` (str): Categorical time string ("morning", "afternoon", "night").
- `weather` (str): Categorical weather condition ("Simulated").
- `cloud_coverage` (float): Real-time cloud fraction (0.0 to 1.0) impacting solar yield.
- `solar_generation` (float): Megawatts of solar energy being actively generated.
- `total_demand` (float): Dynamic total power requirement of the neighborhood grid.
- `battery_charge` (float): Current stored energy available.
- `battery_health` (float): Long-term battery degradation.
- `battery_efficiency` (float): Charge/Discharge loss ratio.

### Action Space
The agent responds with a single integer (0-3) representing the `EnergyAction`:
- **0 - Store:** Route active solar generation into the battery array.
- **1 - Distribute:** Route active solar generation directly to neighborhood demand.
- **2 - Use Battery:** Discharge stored battery energy to meet neighborhood demand.
- **3 - Reduce Load:** Emergency protocol; enforce 20% grid blackout to survive shortages.

---

## 🎯 Task Descriptions & Difficulty

The environment exposes 3 dynamically graded tasks reachable via the `/reset` endpoint. Graders return a `score` between `0.0` and `1.0` evaluating grid survival and energy waste minimization.

### 1. `easy` (Basic Balancing)
- **Difficulty:** Low
- **Description:** A standard 24-hour cycle starting with a heavily charged battery (150kWh). The agent simply needs to maintain a positive balance and not cause arbitrary blackouts.

### 2. `medium` (Midday Demand Spike)
- **Difficulty:** Moderate
- **Description:** Forces a 1.3x neighborhood demand multiplier during the afternoon. The agent must pre-store enough solar energy in the morning to survive the intense afternoon utilization spike.

### 3. `hard` (Nighttime Crisis)
- **Difficulty:** High
- **Description:** The simulation forcefully begins at "night" (0 solar generation) with an ultra-low starting battery charge (20kWh). The agent must aggressively use emergency load reduction without plummeting into total grid collapse.

---

## 🚀 Setup & Usage Instructions

### Docker Execution
This environment is containerized. To spin it up locally:
```bash
docker build -t openenv-solar .
docker run -p 7860:7860 openenv-solar
```

### API Interaction
The Hugging Face Space automatically mounts the required `/reset`, `/step`, and `/state` OpenAPI routes on port `7860`.

```bash
# Reset the environment (Task Selection)
curl -X POST "http://localhost:7860/reset" -H "Content-Type: application/json" -d '{"task_name": "hard"}'

# Submit an action
curl -X POST "http://localhost:7860/step" -H "Content-Type: application/json" -d '{"action": 1}'
```

### Dashboard Access
Navigating to the root domain (`http://localhost:7860/`) provides a full-featured Gradio Dashboard for visual, human-friendly sandbox monitoring and Meta Llama 3 XAI (Explainable AI) auditing.

---

## 📈 Baseline Inference Scores

The repository includes an `inference.py` script standardizing interaction via the OpenAI API client (`OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`).

**Baseline Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
- **Easy Task Score:** ~0.50 (Successfully maintains grid stability via dominant distribution heuristic)
- **Medium Task Score:** ~0.45 
- **Hard Task Score:** ~0.20 (Model struggles to prioritize aggressive short-term load reduction)