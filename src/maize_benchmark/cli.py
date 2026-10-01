from __future__ import annotations

import argparse
from pathlib import Path

from maize_benchmark.config import load_benchmark_config, resolve_project_root


def main() -> None:
    parser = argparse.ArgumentParser(prog="maize-benchmark")
    parser.add_argument("--project-root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--algorithm", required=True, choices=["ppo", "crossq", "dsac_t", "dreamerv3"])
    train.add_argument("--scenario", required=True)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--timesteps", type=int)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--algorithm",
        required=True,
        choices=[
            "ppo",
            "crossq",
            "dsac_t",
            "dreamerv3",
            "rainfed",
            "soil_moisture_threshold",
            "fixed_interval",
        ],
    )
    evaluate.add_argument("--scenario", required=True)
    evaluate.add_argument("--seed", required=True, type=int)
    evaluate.add_argument("--episodes-per-year", type=int)

    sub.add_parser("report")
    args = parser.parse_args()
    root = resolve_project_root(args.project_root)
    benchmark = load_benchmark_config(root)

    if args.command == "report":
        from maize_benchmark.reporting.report import generate_report

        for name, path in generate_report(root).items():
            print(f"{name}: {path}")
        return

    episodes = int(
        getattr(args, "episodes_per_year", None)
        or benchmark["protocol"].get("eval_episodes_per_year", 1)
    )
    if args.command == "train":
        timesteps = int(args.timesteps or benchmark["protocol"]["total_timesteps"])
        if args.algorithm in {"ppo", "crossq"}:
            from maize_benchmark.algorithms.sb3_runner import train_sb3

            train_sb3(
                project_root=root,
                algorithm=args.algorithm,
                scenario=args.scenario,
                seed=args.seed,
                total_timesteps=timesteps,
            )
        elif args.algorithm == "dsac_t":
            from maize_benchmark.algorithms.dsac_t import train_dsac_t

            train_dsac_t(
                project_root=root,
                scenario=args.scenario,
                seed=args.seed,
                total_timesteps=timesteps,
            )
        else:
            from maize_benchmark.algorithms.dreamer_runner import train_dreamerv3

            train_dreamerv3(
                project_root=root,
                scenario=args.scenario,
                seed=args.seed,
                total_timesteps=timesteps,
            )
        return

    if args.algorithm in {"ppo", "crossq"}:
        from maize_benchmark.algorithms.sb3_runner import evaluate_sb3

        evaluate_sb3(
            project_root=root,
            algorithm=args.algorithm,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
        )
    elif args.algorithm == "dsac_t":
        from maize_benchmark.algorithms.dsac_t import evaluate_dsac_t

        evaluate_dsac_t(
            project_root=root,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
        )
    elif args.algorithm == "dreamerv3":
        from maize_benchmark.algorithms.dreamer_runner import evaluate_dreamerv3

        evaluate_dreamerv3(
            project_root=root,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
        )
    else:
        from maize_benchmark.baselines.runner import evaluate_baseline

        evaluate_baseline(
            project_root=root,
            baseline=args.algorithm,
            scenario=args.scenario,
            seed=args.seed,
            episodes_per_year=episodes,
        )


if __name__ == "__main__":
    main()
