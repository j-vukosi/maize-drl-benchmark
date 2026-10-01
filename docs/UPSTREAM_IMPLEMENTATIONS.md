# Upstream Implementations and Benchmark Backends

## AquaCropGymnasium / PPO

Supplied repository: https://github.com/alkaffulm/aquacropgymnasium

The benchmark clones this repository for the climate archive and provenance. The common environment in `src/maize_benchmark/envs/maize_env.py` preserves the source 26-feature observation and AquaCrop stepping logic while adding fixed year lists, a shared normalized action mapped to continuous irrigation depth, seasonal allocation caps, richer metrics and a corrected incremental-water reward option. PPO uses Stable-Baselines3.

## CrossQ

Supplied official repository: https://github.com/adityab/CrossQ

The official release is JAX-based and specifies Python 3.11.5. The Windows-native primary benchmark uses SB3-Contrib CrossQ, a documented implementation of the same ICLR 2024 method, because it shares the same PyTorch/Gymnasium interface as PPO and the benchmark environment. The report must name this backend. The official repository is cloned and its commit is recorded for audit.

## DSAC-T / DSAC-v2

Supplied repository: https://github.com/Jingliang-Duan/DSAC-v2

`src/maize_benchmark/algorithms/dsac_t.py` implements the official DSAC-T refinements in a compact Gymnasium trainer: twin distributional critics, mean-based clipped double-Q selection, sampled return targets, running-standard-deviation TD bounds, variance-aware critic weighting, delayed policy updates, automatic entropy tuning and soft target updates. It is no longer the prototype's simplified “DSAC-T-style” placeholder.

A final dissertation should include a code-equation audit against the recorded upstream commit and report any intentional engineering differences.

## DreamerV3

Supplied official repository: https://github.com/danijar/dreamerv3

The current upstream repository requires Python 3.11+ and states that it is tested on Linux and Mac. The primary Windows-native track therefore uses Ray RLlib DreamerV3 in a separate Python environment. This is an algorithm-equivalent backend, not a claim that the Danijar JAX code runs natively on Windows.

For a higher-assurance replication, run the upstream Danijar repository under WSL2/Linux and compare it as a backend sensitivity analysis. Never combine RLlib and upstream results under one unlabeled “DreamerV3” column.

## Commit provenance

Run:

```powershell
.\scripts\bootstrap_upstreams.ps1
```

The exact commit hashes are written to `outputs/manifests/upstream_commits.json`. Copy this file into the final project appendix.
