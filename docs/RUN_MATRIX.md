# Experiment Run Matrix

## Primary honours preset

| Component | Count |
|---|---:|
| Learned training runs | 4 algorithms × 5 scenarios × 5 seeds = 100 |
| Learned held-out evaluations | 100 |
| Deterministic baseline evaluations | 3 baselines × 5 scenarios × 1 reference seed = 15 |
| Real environment steps per learned run | 512,000 |
| Total planned learned interaction | 51,200,000 AquaCrop steps |

Smoke artifacts are isolated under `outputs_smoke/`; honours artifacts are written under `outputs/`. Each deterministic baseline is evaluated once per scenario to avoid pseudo-replication. The inferential tests in `generate_report.py` include the four DRL algorithms only.

## Resume policy

Run with `-SkipExisting` after interruption. A learned run is skipped only when both its held-out `evaluation.csv` and `interaction_budget.json` exist, the recorded target equals the active preset, and actual environment steps meet that target. Failed processes are recorded in the active output root's `manifests/failures.csv`; do not delete a failed seed and replace it with another seed.

## Validation workflow

Before opening final test results, train a candidate configuration and evaluate it with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --algorithm ppo --scenario normal --seed 11 --split validation
```

Validation output is stored as `validation_evaluation.csv` and is deliberately ignored by the final report collector. Freeze configuration files in version control before executing `--split test`.
