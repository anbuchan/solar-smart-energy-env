---
title: Solar Smart Energy Command Center
emoji: ⚡
colorFrom: yellow
colorTo: blue
sdk: docker
app_file: app.py
app_port: 7860
python_version: "3.10"
tags:
  - openenv
pinned: false
---

# ⚡ AI Smart Solar Energy Command Center
**The production-grade RL environment for Distributed Energy Resource (DER) management.**

This environment strictly follows the **Meta OpenEnv Round 1** specifications. It simulates a high-stakes, real-world energy grid balancing task where an AI agent must manage solar generation, community battery storage, and priority residential/critical loads. This environment models real-world Distributed Energy Resource (DER) optimization under uncertainty and critical infrastructure constraints.

## 🚀 Environment Overview
Grid operators face extreme volatility in renewable generation. This environment enables research into autonomous controllers that can prevent hospital blackouts and optimize grid stability under stormy weather conditions.

## 🎮 Action & Observation Spaces
### Action Space (Discrete)
- `0`: `store_energy` → Direct all current solar into the battery bank.
- `1`: `distribute_energy` → Route energy to homes and hospitals based on demand.
- `2`: `reduce_load` → Demand Response request (reduces residential load by 50%).
- `3`: `prioritize_critical` → **Emergency Protocol**: Zero power to houses; all reserves are dedicated to the Hospital.

### Observation Space
A 12-feature vector tracking: `Time`, `Solar Gen`, `Total Demand`, `Hospital Load`, `Battery SOC`, `Weather Status (Raining)`, and `Per-House Demand Profile`.

## 📈 Baseline Performance Scores
Based on the `Llama-3-8B-Instruct` model in the root `inference.py`:

| Task | Objective | Expected Baseline Score |
| :--- | :--- | :--- |
| **Easy** | Stable solar, baseline residential demand | **0.95** |
| **Medium** | Stormy/Fluctuating weather, predictive storage | **0.82** |
| **Hard** | **Critical Hospital Zone active**, low battery, peak demand | **0.71** |

## 🛠️ Setup & Local Evaluation
1. **Pull Dependencies**: `pip install -r requirements.txt`
2. **Launch Dashboard**: `python app.py` (Tesla-style real-time visualization)
3. **Run Grader**: `python inference.py` (Strict stdout logging for judges)

## 🐳 Containerized Execution
Build and run the OpenEnv compliant container locally:
```bash
docker build -t openenv-solar .
docker run -p 7860:7860 -e HF_TOKEN="your_token" openenv-solar
```

---
*Developed for the Meta OpenEnv Hackathon 2026.*