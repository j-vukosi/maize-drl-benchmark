from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maize_benchmark.config import load_benchmark_config
from maize_benchmark.reporting.statistics import (
    METRIC_DIRECTIONS,
    bootstrap_mean_ci,
    friedman_table,
    pairwise_wilcoxon_table,
    pareto_frequency,
)


DISPLAY_NAMES = {
    "ppo": "PPO",
    "crossq": "CrossQ",
    "dsac_t": "DSAC-T",
    "dreamerv3": "DreamerV3",
    "rainfed": "Rainfed",
    "soil_moisture_threshold": "SMT heuristic",
    "fixed_interval": "Fixed interval",
}


def collect_evaluations(root: Path, output_root: str | Path | None = None) -> pd.DataFrame:
    output = root / (output_root or "outputs")
    files = sorted((output / "runs").glob("*/*/seed_*/evaluation.csv"))
    if not files:
        raise FileNotFoundError(f"No evaluation.csv files found under {output / 'runs'}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        try:
            source_file = path.relative_to(root)
        except ValueError:
            source_file = path
        frame["source_file"] = str(source_file)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _summary(frame: pd.DataFrame, metrics: list[str], resamples: int, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (algorithm, scenario), group in frame.groupby(["algorithm", "scenario"]):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            ci = bootstrap_mean_ci(values, resamples=resamples, alpha=alpha)
            rows.append(
                {
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "metric": metric,
                    "n_independent_seeds": int(values.notna().sum()),
                    "mean": ci.mean,
                    "std": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                    "direction": METRIC_DIRECTIONS.get(metric, "max"),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric(summary: pd.DataFrame, metric: str, algorithms: list[str], outdir: Path) -> None:
    data = summary[(summary["metric"] == metric) & summary["algorithm"].isin(algorithms)]
    if data.empty:
        return
    scenarios = sorted(data["scenario"].unique())
    x = np.arange(len(scenarios), dtype=float)
    width = 0.8 / max(1, len(algorithms))
    fig, ax = plt.subplots(figsize=(13, 7))
    for index, algorithm in enumerate(algorithms):
        subset = data[data["algorithm"] == algorithm].set_index("scenario").reindex(scenarios)
        means = subset["mean"].to_numpy(dtype=float)
        lower = means - subset["ci_lower"].to_numpy(dtype=float)
        upper = subset["ci_upper"].to_numpy(dtype=float) - means
        positions = x - 0.4 + width / 2 + index * width
        ax.bar(positions, means, width=width, label=DISPLAY_NAMES.get(algorithm, algorithm))
        ax.errorbar(positions, means, yerr=np.vstack([lower, upper]), fmt="none", capsize=3)
    ax.set_xticks(x, [name.replace("_", " ") for name in scenarios])
    ax.set_xlabel("Scenario")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').title()} by scenario (95% bootstrap CI)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{metric}_comparison.png", dpi=300)
    plt.close(fig)



def _scenario_design_outputs(root: Path, outdir: Path, summaries_dir: Path) -> None:
    records = []
    groups = []
    labels = []
    for scenario_dir in sorted((root / "scenarios").iterdir()):
        if not scenario_dir.is_dir():
            continue
        manifest_path = scenario_dir / "climate_manifest.csv"
        if not manifest_path.exists():
            continue
        frame = pd.read_csv(manifest_path)
        if frame.empty or "seasonal_precip_mm" not in frame:
            continue
        scenario = scenario_dir.name
        for row in frame.to_dict(orient="records"):
            records.append(
                {
                    "scenario": scenario,
                    "year": int(row["year"]),
                    "split": row.get("split"),
                    "seasonal_precip_mm": float(row["seasonal_precip_mm"]),
                    "seasonal_et0_mm": float(row.get("seasonal_et0_mm", np.nan)),
                }
            )
        groups.append(frame["seasonal_precip_mm"].to_numpy(dtype=float))
        labels.append(scenario.replace("_", " "))
    if not records:
        return
    pd.DataFrame(records).to_csv(summaries_dir / "scenario_year_split.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_ylabel("Seasonal precipitation (mm)")
    ax.set_xlabel("Scenario")
    ax.set_title("Historical climate years used by the five benchmark scenarios")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(outdir / "scenario_precipitation_design.png", dpi=300)
    plt.close(fig)

def _learning_summary(output: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((output / "runs").glob("*/*/seed_*")):
        algorithm, scenario, seed_name = run_dir.parts[-3:]
        seed = int(seed_name.removeprefix("seed_"))
        path = run_dir / "training_episodes.csv"
        if path.exists():
            frame = pd.read_csv(path)
            x_col, y_col = "timesteps", "episode_reward"
        else:
            path = run_dir / "training_iterations.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            x_col, y_col = "environment_steps", "episode_return_mean"
        if frame.empty or x_col not in frame or y_col not in frame:
            continue
        clean = frame[[x_col, y_col]].dropna().sort_values(x_col)
        if clean.empty:
            continue
        rolling = clean[y_col].rolling(window=min(10, len(clean)), min_periods=1).mean()
        x = clean[x_col].to_numpy(dtype=float)
        y = rolling.to_numpy(dtype=float)
        if len(x) > 1 and x[-1] > x[0]:
            auc = float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)) / (x[-1] - x[0]))
        else:
            auc = float(y[-1])
        rows.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "seed": seed,
                "observed_environment_steps": float(x[-1]),
                "learning_curve_auc_mean_reward": auc,
                "final_rolling_reward": float(y[-1]),
                "logged_points": len(clean),
            }
        )
    return pd.DataFrame(rows)


def _compute_summary(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((output / "runs").glob("*/*/seed_*/compute.json")):
        algorithm, scenario, seed_name = path.parts[-4:-1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "seed": int(seed_name.removeprefix("seed_")),
                **payload,
            }
        )
    return pd.DataFrame(rows)




def _budget_summary(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(
        (output / "runs").glob("*/*/seed_*/interaction_budget.json")
    ):
        algorithm, scenario, seed_name = path.parts[-4:-1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "seed": int(seed_name.removeprefix("seed_")),
                **payload,
            }
        )
    return pd.DataFrame(rows)


def _load_execution_plan(output: Path, benchmark: dict[str, Any]) -> dict[str, Any]:
    path = output / "manifests" / "execution_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    protocol = benchmark["protocol"]
    return {
        "preset": "honours-config-default",
        "target_environment_steps": int(protocol["total_timesteps"]),
        "seeds": [int(seed) for seed in protocol["seeds"]],
        "scenarios": list(benchmark["scenarios"]),
        "algorithms": list(benchmark["algorithms"]),
        "baselines": list(benchmark.get("baselines", [])),
    }


def _run_completeness(
    root: Path, output: Path, benchmark: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit every expected run instead of silently dropping missing cells."""

    plan = _load_execution_plan(output, benchmark)
    seeds = [int(seed) for seed in plan.get("seeds", benchmark["protocol"]["seeds"])]
    scenarios = list(plan.get("scenarios", benchmark["scenarios"]))
    algorithms = list(plan.get("algorithms", benchmark["algorithms"]))
    baselines = list(plan.get("baselines", benchmark.get("baselines", [])))
    baseline_seeds = [
        int(seed) for seed in plan.get("baseline_seeds", seeds[:1])
    ]
    target_steps = int(
        plan.get("target_environment_steps", benchmark["protocol"]["total_timesteps"])
    )
    episodes_per_year = int(benchmark["protocol"].get("eval_episodes_per_year", 1))

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        expected_years_path = root / "scenarios" / scenario / "test_years.json"
        expected_years = (
            sorted(int(v) for v in json.loads(expected_years_path.read_text(encoding="utf-8")))
            if expected_years_path.exists()
            else []
        )
        expected_rows = len(expected_years) * episodes_per_year
        for algorithm in [*algorithms, *baselines]:
            learned = algorithm in algorithms
            run_seeds = seeds if learned else baseline_seeds
            for seed in run_seeds:
                run_dir = output / "runs" / algorithm / scenario / f"seed_{seed}"
                evaluation_path = run_dir / "evaluation.csv"
                evaluation_exists = evaluation_path.exists()
                observed_rows = 0
                observed_years: list[int] = []
                evaluation_read_error = ""
                if evaluation_exists:
                    try:
                        evaluation = pd.read_csv(evaluation_path)
                        observed_rows = len(evaluation)
                        if "year" in evaluation:
                            observed_years = sorted(
                                int(v) for v in evaluation["year"].dropna().unique()
                            )
                    except Exception as exc:  # Preserve the failed cell in the audit.
                        evaluation_read_error = repr(exc)

                budget_path = run_dir / "interaction_budget.json"
                budget_exists = budget_path.exists()
                budget_target = budget_actual = budget_overshoot = np.nan
                budget_ok = not learned
                budget_read_error = ""
                if learned and budget_exists:
                    try:
                        payload = json.loads(budget_path.read_text(encoding="utf-8"))
                        budget_target = int(payload["target_environment_steps"])
                        budget_actual = int(payload["actual_environment_steps"])
                        budget_overshoot = int(payload.get("overshoot_steps", 0))
                        budget_ok = budget_target == target_steps and budget_actual >= target_steps
                    except Exception as exc:
                        budget_read_error = repr(exc)

                years_ok = observed_years == expected_years
                rows_ok = observed_rows == expected_rows
                complete = bool(
                    evaluation_exists
                    and not evaluation_read_error
                    and years_ok
                    and rows_ok
                    and budget_ok
                )
                rows.append(
                    {
                        "preset": plan.get("preset", "unknown"),
                        "algorithm": algorithm,
                        "scenario": scenario,
                        "seed": seed,
                        "learned_agent": learned,
                        "evaluation_exists": evaluation_exists,
                        "expected_test_years": ";".join(map(str, expected_years)),
                        "observed_test_years": ";".join(map(str, observed_years)),
                        "expected_evaluation_rows": expected_rows,
                        "observed_evaluation_rows": observed_rows,
                        "budget_metadata_exists": budget_exists if learned else np.nan,
                        "expected_target_environment_steps": target_steps if learned else np.nan,
                        "recorded_target_environment_steps": budget_target,
                        "actual_environment_steps": budget_actual,
                        "overshoot_steps": budget_overshoot,
                        "evaluation_read_error": evaluation_read_error,
                        "budget_read_error": budget_read_error,
                        "complete": complete,
                    }
                )

    frame = pd.DataFrame(rows)
    learned_frame = frame[frame["learned_agent"]] if not frame.empty else frame
    all_learned_complete = bool(
        len(learned_frame) > 0 and learned_frame["complete"].all()
    )
    honours_protocol_matches = bool(
        plan.get("preset") == "honours"
        and target_steps == int(benchmark["protocol"]["total_timesteps"])
        and sorted(seeds) == sorted(int(v) for v in benchmark["protocol"]["seeds"])
        and sorted(scenarios) == sorted(benchmark["scenarios"])
        and sorted(algorithms) == sorted(benchmark["algorithms"])
    )
    status = {
        "preset": plan.get("preset", "unknown"),
        "output_root": str(output.relative_to(root) if output.is_relative_to(root) else output),
        "expected_learned_runs": int(len(learned_frame)),
        "complete_learned_runs": int(learned_frame["complete"].sum()) if not frame.empty else 0,
        "all_expected_learned_runs_complete": all_learned_complete,
        "honours_protocol_matches_frozen_config": honours_protocol_matches,
        "valid_for_final_ranking": bool(all_learned_complete and honours_protocol_matches),
        "expected_reference_runs": int((~frame["learned_agent"]).sum()) if not frame.empty else 0,
        "complete_reference_runs": int(
            frame.loc[~frame["learned_agent"], "complete"].sum()
        )
        if not frame.empty
        else 0,
    }
    return frame, status

def generate_report(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    benchmark = load_benchmark_config(root)
    output = root / benchmark["output_root"]
    evaluations_dir = output / "evaluations"
    summaries_dir = output / "summaries"
    statistics_dir = output / "statistics"
    figures_dir = output / "figures"
    for folder in [evaluations_dir, summaries_dir, statistics_dir, figures_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    _scenario_design_outputs(root, figures_dir, summaries_dir)

    completeness, completeness_status = _run_completeness(root, output, benchmark)
    completeness.to_csv(summaries_dir / "run_completeness.csv", index=False)
    write_target = output / "manifests" / "completion_status.json"
    write_target.parent.mkdir(parents=True, exist_ok=True)
    write_target.write_text(
        json.dumps(completeness_status, indent=2, sort_keys=True), encoding="utf-8"
    )

    all_rows = collect_evaluations(root, benchmark["output_root"])
    # Detailed deterministic evaluation: one row per algorithm × seed × held-out year.
    primary = all_rows.sort_values("episode").drop_duplicates(
        ["algorithm", "scenario", "seed", "year"], keep="first"
    )
    all_rows.to_csv(evaluations_dir / "all_evaluation_rows.csv", index=False)
    primary.to_csv(evaluations_dir / "primary_seed_year_rows.csv", index=False)

    metrics = [metric for metric in METRIC_DIRECTIONS if metric in primary.columns]
    # Inferential unit: one independently trained policy seed, averaged across the
    # shared held-out climate years. Seed-year rows remain available for climate-
    # specific inspection and the descriptive Pareto-frequency calculation.
    seed_aggregate = (
        primary.groupby(["algorithm", "scenario", "seed"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    seed_aggregate.to_csv(
        evaluations_dir / "primary_seed_aggregate_rows.csv", index=False
    )

    protocol = benchmark["protocol"]
    summary = _summary(
        seed_aggregate,
        metrics,
        int(protocol.get("bootstrap_resamples", 10000)),
        float(protocol.get("primary_alpha", 0.05)),
    )
    summary.to_csv(summaries_dir / "algorithm_scenario_summary.csv", index=False)

    algorithms = list(benchmark["algorithms"])
    friedman = friedman_table(seed_aggregate, metrics, algorithms)
    pairwise = pairwise_wilcoxon_table(seed_aggregate, metrics, algorithms)
    pareto = pareto_frequency(primary, algorithms)
    friedman.to_csv(statistics_dir / "friedman_tests.csv", index=False)
    pairwise.to_csv(statistics_dir / "pairwise_wilcoxon_holm.csv", index=False)
    pareto.to_csv(statistics_dir / "yield_water_pareto_frequency.csv", index=False)

    learning = _learning_summary(output)
    compute = _compute_summary(output)
    budget = _budget_summary(output)
    learning.to_csv(summaries_dir / "learning_efficiency_summary.csv", index=False)
    compute.to_csv(summaries_dir / "compute_summary.csv", index=False)
    budget.to_csv(summaries_dir / "interaction_budget_summary.csv", index=False)
    for metric in metrics:
        _plot_metric(summary, metric, algorithms, figures_dir)

    return {
        "all_rows": evaluations_dir / "all_evaluation_rows.csv",
        "summary": summaries_dir / "algorithm_scenario_summary.csv",
        "friedman": statistics_dir / "friedman_tests.csv",
        "pairwise": statistics_dir / "pairwise_wilcoxon_holm.csv",
        "figures": figures_dir,
    }
