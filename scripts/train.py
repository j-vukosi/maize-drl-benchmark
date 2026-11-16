#!/usr/bin/env python3
"""Train one benchmark policy using the frozen benchmark configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from maize_benchmark.config import resolve_project_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "crossq", "dsac_t", "dreamerv3"])
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    from maize_benchmark.config import load_benchmark_config

    config = load_benchmark_config(project_root)
    timesteps = int(args.timesteps or config["protocol"]["total_timesteps"])
    if timesteps <= 0:
        parser.error("--timesteps must be a positive integer")

    if args.algorithm in {"ppo", "crossq"}:
        from maize_benchmark.algorithms.sb3_runner import train_sb3

        output = train_sb3(
            project_root=project_root,
            algorithm=args.algorithm,
            scenario=args.scenario,
            seed=args.seed,
            total_timesteps=timesteps,
        )
    elif args.algorithm == "dsac_t":
        from maize_benchmark.algorithms.dsac_t import train_dsac_t

        output = train_dsac_t(
            project_root=project_root,
            scenario=args.scenario,
            seed=args.seed,
            total_timesteps=timesteps,
        )
    else:
        from maize_benchmark.algorithms.dreamer_runner import train_dreamerv3

        output = train_dreamerv3(
            project_root=project_root,
            scenario=args.scenario,
            seed=args.seed,
            total_timesteps=timesteps,
        )
    print(output)


if __name__ == "__main__":
    main()
