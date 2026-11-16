#!/usr/bin/env python3
"""Evaluate a learned policy or conventional baseline on held-out years."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from maize_benchmark.config import load_benchmark_config, resolve_project_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[
            "ppo", "crossq", "dsac_t", "dreamerv3",
            "rainfed", "soil_moisture_threshold", "fixed_interval",
        ],
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes-per-year", type=int)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    benchmark = load_benchmark_config(project_root)
    episodes = int(
        args.episodes_per_year
        or benchmark["protocol"].get("eval_episodes_per_year", 1)
    )
    if episodes <= 0:
        parser.error("--episodes-per-year must be positive")

    if args.algorithm in {"ppo", "crossq"}:
        from maize_benchmark.algorithms.sb3_runner import evaluate_sb3

        output = evaluate_sb3(
            project_root=project_root,
            algorithm=args.algorithm,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
            split=args.split,
        )
    elif args.algorithm == "dsac_t":
        from maize_benchmark.algorithms.dsac_t import evaluate_dsac_t

        output = evaluate_dsac_t(
            project_root=project_root,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
            split=args.split,
        )
    elif args.algorithm == "dreamerv3":
        from maize_benchmark.algorithms.dreamer_runner import evaluate_dreamerv3

        output = evaluate_dreamerv3(
            project_root=project_root,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
            split=args.split,
        )
    else:
        from maize_benchmark.baselines.runner import evaluate_baseline

        output = evaluate_baseline(
            project_root=project_root,
            baseline=args.algorithm,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
            split=args.split,
        )
    print(output)


if __name__ == "__main__":
    main()
