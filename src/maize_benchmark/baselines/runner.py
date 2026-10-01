from __future__ import annotations

from pathlib import Path

from maize_benchmark.baselines.policies import make_baseline
from maize_benchmark.config import load_benchmark_config
from maize_benchmark.evaluation import evaluate_policy, save_evaluation


def evaluate_baseline(
    *,
    project_root: str | Path,
    baseline: str,
    scenario: str,
    seed: int,
    episodes_per_year: int,
    split: str = "test",
) -> Path:
    root = Path(project_root)
    benchmark = load_benchmark_config(root)
    frame = evaluate_policy(
        project_root=root,
        algorithm=baseline,
        scenario=scenario,
        seed=seed,
        policy=make_baseline(baseline),
        episodes_per_year=episodes_per_year,
        deterministic=True,
        split=split,
    )
    target = (
        root
        / benchmark["output_root"]
        / "runs"
        / baseline
        / scenario
        / f"seed_{seed}"
        / ("evaluation.csv" if split == "test" else "validation_evaluation.csv")
    )
    save_evaluation(frame, target)
    return target
