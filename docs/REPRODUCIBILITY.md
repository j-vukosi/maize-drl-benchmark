# Reproducibility Record

Every run directory contains:

- `resolved_config.json`;
- `system_manifest.json` with Python, OS, CPU/GPU and package freeze;
- training logs;
- model/checkpoint files;
- `compute.json`;
- `interaction_budget.json`;
- `evaluation.csv`.

Global manifests contain:

- climate-file SHA-256;
- exact scenario years and split;
- upstream Git commits;
- limited-water cap calibration;
- preset-specific `execution_plan.json`;
- `completion_status.json` and `run_completeness.csv`.

Do not overwrite a completed run. Copy the configuration, add an experiment identifier and write to a new output root. Archive the full `outputs/manifests` folder with the report.
