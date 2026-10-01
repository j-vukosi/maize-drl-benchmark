# Build Validation and Known Runtime Boundary

## Validation completed before delivery

The package was checked without inventing benchmark outcomes:

- every Python module compiles successfully;
- all unit tests pass (`10 passed` at delivery time), including the normalized-action round trip;
- scenario selection and deterministic year splitting were executed against the supplied `champion_climate.txt` archive;
- an end-to-end reporting dry run was completed with synthetic rows in a temporary folder to verify CSV aggregation, confidence intervals, paired tests, output-root isolation, completeness auditing and figure generation;
- the generated scenario manifests were inspected for disjoint train/validation/test years;
- the DSAC-T update test exercises actor, twin distributional critics, target networks and entropy temperature;
- the DreamerV3 inference adapter was audited against Ray 2.48's RLModule API, including recurrent state and `is_first` handling.

## What was not executed in the delivery sandbox

The delivery environment did not contain AquaCrop, Stable-Baselines3, SB3-Contrib, Ray or TensorFlow and had no network package installation. Therefore, a genuine AquaCrop training run and the RLlib DreamerV3 runtime were not executed here. The Windows bootstrap scripts install those dependencies and `scripts/validate_install.py` then runs the Gymnasium/SB3 environment checker plus a real AquaCrop reset/step smoke test before training.

Do not report final algorithm rankings until the target Windows machine has completed the multi-seed honours preset. A smoke run proves integration only.

## First-machine acceptance gates

A target machine is ready only when all of the following succeed:

```powershell
.\scripts\bootstrap_windows.ps1
.\scripts\bootstrap_dreamer_windows.ps1
.\scripts\run_benchmark.ps1 -Preset smoke
.\.venv\Scripts\python.exe -m pytest
```

Confirm that every smoke run under `outputs_smoke/runs/` contains `evaluation.csv`, `resolved_config.json`, `system_manifest.json`, a model/checkpoint, a training log, `compute.json` and `interaction_budget.json`. The smoke completion status must remain `valid_for_final_ranking: false`. After the honours preset, confirm `outputs/manifests/completion_status.json` reports `valid_for_final_ranking: true` before interpreting final tables. Failures are appended to the active output root's `manifests/failures.csv`.
