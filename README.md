<div align="center">

# 🌽 Maize DRL Benchmark

### Deep Reinforcement Learning for Adaptive Maize Irrigation Scheduling

*A reproducible benchmark using AquaCropGymnasium*

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square\&logo=python\&logoColor=white)](https://www.python.org/)
[![RL](https://img.shields.io/badge/Deep%20RL-Benchmark-8A2BE2?style=flat-square)](#)
[![AquaCrop](https://img.shields.io/badge/AquaCrop-Simulation-4CAF50?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Research-success?style=flat-square)](#)

</div>

---

## Overview

This project implements a **reproducible benchmark of four deep reinforcement learning algorithms** for adaptive maize irrigation scheduling:

* **PPO**
* **CrossQ**
* **DSAC-T**
* **DreamerV3**

All algorithms are evaluated within a common **AquaCropGymnasium** environment under the same benchmark scenarios, irrigation formulation, evaluation framework, and interaction budget.

The benchmark examines:

**🌽 Yield · 💧 Irrigation Water · 💰 Profit · 📈 Learning Behaviour · 💻 Computational Cost**

---

## 🔬 Research Question

> Which deep reinforcement learning approach provides a reliable balance between maize yield protection and water-efficient irrigation across different rainfall and water-availability scenarios?

---

## ⚙️ Environment

The benchmark uses an AquaCrop-based maize environment in which an RL agent observes crop, soil, and weather information and determines irrigation actions throughout the growing season.

```text
Weather + Crop + Soil State
            ↓
      Agent Observation
            ↓
        DRL Policy
            ↓
     Irrigation Action
            ↓
     AquaCrop Simulation
            ↓
   Yield / Water / Reward
```

### Corrected Environment

Before the final benchmark was retrained, two environment issues were audited and corrected.

**Irrigation action**

The continuous action was changed from requesting an absolute irrigation depth to representing the **fraction of current root-zone depletion to refill**.

The action remains subject to:

* **40% TAW depletion trigger**
* **1 mm minimum irrigation event**
* **25 mm maximum daily irrigation**

**Weather alignment**

AquaCrop continues to receive the full annual weather series, while the weather-history observations used by the RL agent are aligned to the crop season beginning **1 May**.

These corrections were applied before the final benchmark experiments.

---

## 🌦️ Benchmark Scenarios

Five scenarios are evaluated:

| Scenario        | Condition                     |
| --------------- | ----------------------------- |
| `normal`        | Baseline seasonal variability |
| `dry`           | Dry conditions                |
| `extreme_dry`   | Severe dry conditions         |
| `limited_water` | Water-use restriction         |
| `wet`           | Higher-rainfall conditions    |

Historical climate years are used rather than a single synthetic average season.

---

## 🧪 Experimental Protocol

| Setting             |                  Value |
| ------------------- | ---------------------: |
| Algorithms          |                      4 |
| Scenarios           |                      5 |
| Independent seeds   |                      5 |
| Seeds               |   `11, 23, 37, 53, 71` |
| Steps per run       |            **204,800** |
| Total trained runs  |                **100** |
| Evaluation          | Held-out climate years |
| Bootstrap resamples |             **10,000** |

All 100 final benchmark runs reached exactly **204,800 environment steps**, with no recorded interaction-budget overshoot.

---

## 📊 Evaluation

Policies are evaluated using:

* **Dry yield** (`t ha⁻¹`)
* **Seasonal irrigation applied** (`mm`)
* **Water productivity**
* **Profit**
* **Episode reward**
* **Learning behaviour**
* **Wall-clock time**
* **Peak memory usage**

A Pareto analysis is also used to examine:

> **Higher yield + lower irrigation**

Uncertainty is reported using **95% bootstrap confidence intervals based on the five independent training seeds**.

---

## 📈 Selected Results

### Mean Dry Yield

| Scenario      |       PPO | CrossQ |    DSAC-T | DreamerV3 |
| ------------- | --------: | -----: | --------: | --------: |
| Extreme Dry   | **13.80** |      — |         — |      4.56 |
| Dry           | **13.34** |      — |         — |         — |
| Normal        | **13.57** |      — |         — |         — |
| Limited Water | **12.83** |      — |         — |         — |
| Wet           |     13.25 |      — | **13.27** |     13.09 |

PPO recorded the highest mean yield in four of the five benchmark scenarios, while DSAC-T recorded the highest mean yield under wet conditions.

### Mean Seasonal Irrigation

| Scenario      | Lowest observed irrigation |
| ------------- | -------------------------- |
| Extreme Dry   | DreamerV3 — **142.60 mm**  |
| Dry           | CrossQ — **133.47 mm**     |
| Normal        | CrossQ — **218.36 mm**     |
| Limited Water | CrossQ — **162.18 mm**     |
| Wet           | CrossQ — **64.85 mm**      |

The results illustrate the trade-off between crop production and irrigation demand.

---

## 💻 Computational Cost

Observed average execution per trained run:

| Algorithm | Wall Time | Peak RSS |
| --------- | --------: | -------: |
| PPO       |    0.52 h | 0.68 GiB |
| CrossQ    |    3.88 h | 0.85 GiB |
| DSAC-T    |    0.70 h | 0.71 GiB |
| DreamerV3 |    0.89 h | 5.98 GiB |

These measurements describe the **implemented benchmark configurations**.

---

## 📁 Project Structure

```text
maize-drl-benchmark/
│
├── configs/
├── docs/
├── external/
│   └── aquacropgymnasium/
│       └── weather_data/
├── requirements/
├── scenarios/
├── src/
├── tests/
├── CITATIONS.bib
├── LICENSE
├── README.md
└── pyproject.toml
```

The required climate data is included in the repository tree.

---

## 🚀 Running

### Smoke Test

A lightweight configuration is available for validating the pipeline:

```text
Timesteps: 5,000
Seed:      11
```

```bash
./run_benchmark.sh
```

### Full Benchmark

```text
Timesteps: 204,800
Seeds:     11 23 37 53 71
```

```bash
PRESET=benchmark ./run_benchmark.sh
```

---

## 🔁 Reproducibility

The benchmark records and controls:

* random seeds
* scenario configuration
* training budgets
* evaluation assignments
* model outputs
* interaction budgets
* computational measurements

The full experimental configuration is defined in:

```text
benchmark.yaml
```

---

## ⚠️ Scope

This is a **simulation-based research benchmark**.

It does not include:

* physical irrigation hardware
* live sensor integration
* field deployment
* real-time irrigation actuation

Results should therefore be interpreted within the configured AquaCropGymnasium environment and benchmark protocol.

---

## 📚 Key References

[1] M. Alkaff, A. Basuhail and Y. Sari, *Optimizing Water Use in Maize Irrigation with Reinforcement Learning*, Mathematics, 2025.

[2] A. Bhatt et al., *CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity*, ICLR, 2024.

[3] J. Duan et al., *Distributional Soft Actor-Critic With Three Refinements*, IEEE TPAMI, 2025.

[4] D. Hafner et al., *Mastering Diverse Control Tasks through World Models*, Nature, 2025.

---

<div align="center">

### 🌱 Reproducible Research · 🤖 Deep Reinforcement Learning · 💧 Smart Irrigation

</div>
