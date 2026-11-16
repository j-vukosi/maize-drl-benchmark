# 🌽 Maize DRL Benchmark

Run **PPO, CrossQ, DSAC-T, and DreamerV3** on a shared AquaCrop simulation of maize irrigation in Champion, Nebraska. The benchmark compares grain yield, applied irrigation, water-use efficiency, and estimated profit.

| Scenario | Training years | Test years | Seeds | Steps per policy |
| --- | --- | --- | --- | ---: |
| `baseline_full` | 1982–2007 | 2008–2018 | `1`, `2`, `3` | 1,500,000 |

The full benchmark trains **12 policies** and evaluates each policy on **11 held-out seasons**.

## 🛠️ Install and fetch the weather data

Use Linux with Bash, Python 3.11, Git, and curl. Run the commands from the repository root after cloning.

```bash
git clone https://github.com/j-vukosi/maize-drl-benchmark.git
cd maize-drl-benchmark

python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/base.txt

weather_dir=external/aquacropgymnasium/weather_data
mkdir -p "$weather_dir"
curl -fL 'https://raw.githubusercontent.com/alkaffulm/aquacropgymnasium/19bf133fb12f2587c8df99bc27eaf053e0e76c89/weather_data/champion_climate.txt' \
  -o "$weather_dir/champion_climate.txt"
```

<details>
<summary>Install DreamerV3 dependencies</summary>

Complete this step before running DreamerV3 or the full benchmark.

```bash
python3.11 -m venv .venv_dreamer
.venv_dreamer/bin/python -m pip install -r requirements/dreamer.txt "torch>=2.3,<3" "pyarrow==23.0.1"
.venv_dreamer/bin/python -m pip install --no-deps -e .
```

</details>

## ▶️ Train and evaluate a policy

In [`configs/algorithms/crossq.yaml`](configs/algorithms/crossq.yaml), set `learning_starts: 2048` and `batch_size: 512` for CrossQ. The example below trains seed 1 and evaluates its checkpoint on the test years. Archive any existing `outputs/` directory before training if you need to keep previous runs.

```bash
.venv/bin/python scripts/train.py --algorithm crossq --scenario baseline_full --seed 1 --timesteps 1500000 &&
  .venv/bin/python scripts/evaluate.py --algorithm crossq --scenario baseline_full --seed 1 --split test
```

Use `ppo`, `crossq`, `dsac_t`, or `dreamerv3` with `--algorithm`. Use `.venv_dreamer/bin/python` for both DreamerV3 commands. Benchmark and algorithm settings are in [`configs/`](configs/), and the year splits are in [`scenarios/baseline_full/`](scenarios/baseline_full/).

<details>
<summary>Train and evaluate all 12 policies</summary>

This loop runs all three seeds for each algorithm and stops if a command fails.

```bash
(
  set -e
  for algorithm in ppo crossq dsac_t dreamerv3; do
    python_bin=.venv/bin/python
    if [[ "$algorithm" == dreamerv3 ]]; then
      python_bin=.venv_dreamer/bin/python
    fi

    for seed in 1 2 3; do
      "$python_bin" scripts/train.py \
        --algorithm "$algorithm" --scenario baseline_full --seed "$seed" --timesteps 1500000
      "$python_bin" scripts/evaluate.py \
        --algorithm "$algorithm" --scenario baseline_full --seed "$seed" --split test
    done
  done
)
```

</details>

Each run saves its checkpoint, `evaluation.csv`, and `interaction_budget.json` under `outputs/runs/<algorithm>/baseline_full/seed_<seed>/`. To evaluate an existing checkpoint, run only the evaluation command with its algorithm and seed. For PPO, keep `vecnormalize.pkl` alongside the checkpoint.

## 📊 Audit runs and generate reports

After all 12 runs finish, check the interactions recorded for each run, then generate the summaries and figures.

```bash
.venv/bin/python scripts/audit_benchmark.py --output-root outputs --expected-steps 1500000 &&
  .venv/bin/python scripts/generate_report.py &&
  .venv/bin/python scripts/generate_baseline_full_climate_graphs.py
```

| Output | Location |
| --- | --- |
| Benchmark summaries | `outputs/summaries/` |
| Statistical tables | `outputs/statistics/` |
| Rainfall-group figures | `outputs/figures/baseline_full_climate/` |

The figures show **Extreme Dry, Dry, Normal, and Wet**, in that order.

## 📄 Licence

The benchmark code is released under the [MIT License](LICENSE).
