# Windows Setup

## Recommended software

- Windows 10 or 11, 64-bit;
- VS Code with the Python extension;
- Git for Windows;
- Python 3.10 for PPO, CrossQ and DSAC-T;
- Python 3.11 for the separate RLlib DreamerV3 environment;
- at least 16 GB RAM; 32 GB is preferable for full runs;
- an NVIDIA GPU is helpful but not required for the base agents.

Keep the project in an English-only path without cloud-sync locking, for example:

```text
D:\School\maize_drl_benchmark
```

## Automated setup

In PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
cd D:\School\maize_drl_benchmark
.\scripts\bootstrap_windows.ps1
.\scripts\bootstrap_dreamer_windows.ps1
```

The first script clones all supplied upstream repositories, creates `.venv`, installs the base dependencies, creates scenario manifests, calibrates the limited-water cap and validates the installation.

The second script creates `.venv_dreamer` with a pinned Ray/TensorFlow combination. Keeping it separate prevents the Gymnasium and TensorFlow stack from destabilising the SB3 environment.

## Smoke test

```powershell
.\scripts\run_benchmark.ps1 -Preset smoke
```

The smoke preset proves that files, training loops and evaluation outputs work. It is not valid evidence for final algorithm ranking. Its artifacts are stored in `outputs_smoke/`, physically separate from the honours results in `outputs/`.

## Full experiment

```powershell
.\scripts\run_benchmark.ps1 -Preset honours
```

This launches 4 algorithms × 5 scenarios × 5 seeds. It is a large experiment. Use `-SkipExisting` when resuming; the script verifies matching interaction-budget metadata before skipping a learned run.

## Common failures

### Python launcher cannot find a version

```powershell
py -0p
```

Install the missing 64-bit Python version, close VS Code and reopen it.

### PowerShell blocks scripts

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Git is not recognised

Install Git for Windows and ensure “Git from the command line” is selected.

### Dreamer TensorFlow/Keras error

Do not upgrade one Dreamer dependency independently. Recreate the environment:

```powershell
Remove-Item -Recurse -Force .venv_dreamer
.\scripts\bootstrap_dreamer_windows.ps1
```

### Out of memory

Use the smoke preset, reduce Dreamer `batch_size_B`, or run the upstream Dreamer track in WSL2 on a GPU machine. Record any change as a new configuration.
