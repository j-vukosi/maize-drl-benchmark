#!/usr/bin/env python3
"""Audit all configured learned-run evaluations and exact step budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--expected-steps", type=int, default=1500000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    import sys
    sys.path.insert(0, str(root / "src"))
    from maize_benchmark.config import load_benchmark_config

    benchmark = load_benchmark_config(root)
    output_root = args.output_root or benchmark["output_root"]
    run_root = root / output_root / "runs"
    invalid: list[str] = []
    expected_per_algorithm = len(benchmark["scenarios"]) * len(benchmark["protocol"]["seeds"])
    expected_total = expected_per_algorithm * len(benchmark["algorithms"])
    for algorithm in benchmark["algorithms"]:
        evaluations = list(
            (run_root / algorithm / "baseline_full")
            .glob("seed_*/evaluation.csv")
        )
        budgets = list((run_root / algorithm).glob("*/*/interaction_budget.json"))
        print(f"{algorithm:10s} evaluations={len(evaluations):2d} budgets={len(budgets):2d}")
        if len(evaluations) != expected_per_algorithm:
            invalid.append(f"{algorithm}: expected {expected_per_algorithm} evaluation.csv, found {len(evaluations)}")
        if len(budgets) != expected_per_algorithm:
            invalid.append(f"{algorithm}: expected {expected_per_algorithm} interaction_budget.json, found {len(budgets)}")
        for path in budgets:
            data = json.loads(path.read_text(encoding="utf-8"))
            target = int(data.get("target_environment_steps", -1))
            actual = int(data.get("actual_environment_steps", -1))
            overshoot = int(data.get("overshoot_steps", actual - target))
            if target != args.expected_steps or actual != args.expected_steps or overshoot != 0:
                invalid.append(
                    f"{path}: target={target}, actual={actual}, overshoot={overshoot}"
                )

    print(f"INVALID RECORDS: {len(invalid)}")
    for item in invalid:
        print(f"FAIL: {item}")
    if not invalid:
        print(f"PASS: all {expected_total} learned runs have exactly {args.expected_steps} steps")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
