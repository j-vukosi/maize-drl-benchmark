# Canonical Danijar DreamerV3 Track in WSL2

The official repository is tested on Linux/Mac and requires Python 3.11+. Use Ubuntu under WSL2 for this optional replication track.

1. Install WSL2 and an Ubuntu distribution.
2. From Ubuntu, clone the benchmark and the official DreamerV3 repository.
3. Create a Python 3.11 virtual environment.
4. Install JAX for the available CPU/CUDA platform, then the official `dreamerv3/requirements.txt`.
5. Implement an `embodied.Env` adapter that exposes the exact same scaled 26-vector observation, normalized `[-1,1]` action mapped to `[0,25]` mm, reward and reset-year manifests used by the primary benchmark.
6. Freeze the official commit, configuration and JAX/CUDA versions.
7. Train with exactly the same real environment-step budget and evaluate on the same held-out years, then use the same seed-level aggregation and paired analysis.

This repository supplies the Windows-complete RLlib track. The WSL track is an optional independent replication and should be reported separately until its adapter passes trajectory-level equivalence tests against the common environment.
