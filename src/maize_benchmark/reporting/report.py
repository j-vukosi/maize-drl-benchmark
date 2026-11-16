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


SCENARIO_ORDER = ["extreme_dry", "dry", "normal", "limited_water", "wet"]
SCENARIO_LABELS = {
    "extreme_dry": "Extreme Dry",
    "dry": "Dry",
    "normal": "Normal",
    "limited_water": "Limited Water",
    "wet": "Wet",
}

# Contrast-safe palette: error bars remain near-black.
ERRORBAR_COLOR = "#161616"
BAR_COLORS = {
    "ppo": "#2E9F6B",
    "crossq": "#D6B56E",
    "dsac_t": "#7D8590",
    "dreamerv3": "#3B82B8",
}

# SVG is the only figure format produced by this reporting module.
FIGURE_FILENAMES = {
    "dry_yield_tonne_per_ha": "mean_dry_yield_by_scenario.svg",
    "total_irrigation_mm": "total_irrigation_applied_by_scenario.svg",
    "water_productivity_kg_per_ha_per_mm": "water_efficiency_by_scenario.svg",
    "profit": "profit_usd_per_ha_by_scenario.svg",
    "episode_reward": "evaluation_episode_reward_by_scenario.svg",
    "stress_exposure": "crop_water_stress_exposure_by_scenario.svg",
    "stress_days": "crop_water_stress_days_by_scenario.svg",
    "max_daily_stress": "maximum_daily_crop_water_stress_by_scenario.svg",
}

# Reference: Alkaff, Basuhail & Sari (2025), Mathematics 13(4), 595.
# DOI: 10.3390/math13040595
# Baseline reporting units: yield (t/ha), irrigation (mm), water efficiency
# (kg/ha/mm), and modelled profit (USD/ha).
# Baseline economic assumptions, section 4.2.4:
#   crop price = USD 180/tonne
#   irrigation expenditure = USD 1/(ha-mm)
#   fixed non-irrigation costs = USD 1728/ha
# This changes evaluation/reporting metrics only, NOT policy training rewards.
BASELINE_CROP_PRICE_USD_PER_TONNE = 180.0
BASELINE_IRRIGATION_COST_USD_PER_HA_MM = 1.0
BASELINE_FIXED_COST_USD_PER_HA = 1728.0
METRIC_FORMAT = {
    "dry_yield_tonne_per_ha": ("Dry Yield (t/ha)", "Mean dry yield"),
    "total_irrigation_mm": ("Seasonal Irrigation (mm)", "Total irrigation applied"),
    "water_productivity_kg_per_ha_per_mm": (
        "Water Efficiency (kg/ha/mm)", "Irrigation water efficiency"
    ),
    "profit": ("Profit (USD/ha)", "Modelled profitability"),
    "episode_reward": ("Evaluation episode return", "Evaluation return"),
    "stress_exposure": ("Cumulative root-zone depletion stress", "Crop water-stress exposure"),
    "stress_days": ("Stress days (days)", "Crop water-stress days"),
    "max_daily_stress": (
        "Maximum daily water stress (fraction)", "Maximum crop water stress"
    ),
}
DISPLAY_NAMES = {
    "ppo": "PPO",
    "crossq": "CrossQ",
    "dsac_t": "DSAC-T",
    "dreamerv3": "DreamerV3",
    "rainfed": "Rainfed",
    "soil_moisture_threshold": "SMT (4 growth stages)",
    "fixed_interval": "Fixed interval (7 days)",
}


def _save_svg(fig, svg_path: Path) -> None:
    """Save an infinitely scalable SVG only (no PNG output)."""
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg", bbox_inches="tight")


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


def _apply_baseline_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Recalculate baseline metrics from per-episode yield and irrigation.

    Recomputing at the evaluation-row level (before seed averaging) ensures
    all report CSVs, statistical tests and SVG plots use the same formulas.
    Raw outputs/runs/**/evaluation.csv files are deliberately left untouched.
    """
    required = {"dry_yield_tonne_per_ha", "total_irrigation_mm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Cannot apply baseline formulas: evaluation data lacks "
            + ", ".join(sorted(missing))
        )

    result = frame.copy()
    yield_t_ha = pd.to_numeric(
        result["dry_yield_tonne_per_ha"], errors="coerce"
    )
    irrigation_mm = pd.to_numeric(
        result["total_irrigation_mm"], errors="coerce"
    )
    # The baseline formula only has a physical interpretation for non-negative
    # yields and applied irrigation. Keep bad values missing instead of
    # creating plausible-looking results.
    physically_valid = yield_t_ha.ge(0) & irrigation_mm.ge(0)
    yield_t_ha = yield_t_ha.where(physically_valid)
    irrigation_mm = irrigation_mm.where(physically_valid)

    # Alkaff et al. economic model, USD/ha:
    # P = 180 * Y - 1.00 * I - 1728.
    result["profit"] = (
        BASELINE_CROP_PRICE_USD_PER_TONNE * yield_t_ha
        - BASELINE_IRRIGATION_COST_USD_PER_HA_MM * irrigation_mm
        - BASELINE_FIXED_COST_USD_PER_HA
    )

    # Water efficiency: (1000 kg/t * yield t/ha) / irrigation mm.
    # No supplemental irrigation -> undefined (NaN), not zero.
    result["water_productivity_kg_per_ha_per_mm"] = (
        1000.0 * yield_t_ha / irrigation_mm.where(irrigation_mm > 0)
    )
    return result


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

    present = set(data["scenario"].dropna().astype(str))
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in present]
    scenarios.extend(sorted(present.difference(scenarios)))
    x = np.arange(len(scenarios), dtype=float)
    width = 0.84 / max(1, len(algorithms))
    fig, ax = plt.subplots(figsize=(17, 8))
    drawn_limits: list[float] = []

    # Match the baseline figure presentation: no error bars on the
    # water-efficiency and profitability diagrams. Other metrics show 95% CIs.
    metrics_without_error_bars = {
        "water_productivity_kg_per_ha_per_mm",
        "profit",
    }

    for index, algorithm in enumerate(algorithms):
        subset = data[data["algorithm"] == algorithm].set_index("scenario").reindex(scenarios)
        means = subset["mean"].to_numpy(dtype=float)
        lower_ci = subset["ci_lower"].to_numpy(dtype=float)
        upper_ci = subset["ci_upper"].to_numpy(dtype=float)
        count = subset["n_independent_seeds"].to_numpy(dtype=float)
        positions = x - 0.42 + width / 2 + index * width
        valid = np.isfinite(means)

        ax.bar(
            positions[valid], means[valid], width=width * 0.94,
            color=BAR_COLORS.get(algorithm, "#6B7280"),
            edgecolor="#333333", linewidth=0.55,
            label=DISPLAY_NAMES.get(algorithm, algorithm), zorder=2,
        )
        # Bootstrap confidence intervals use independently trained seed means.
        # Do not draw misleading error bars when fewer than two seeds are valid.
        ci_valid = (
            valid
            & (count >= 2)
            & np.isfinite(lower_ci)
            & np.isfinite(upper_ci)
            & (metric not in metrics_without_error_bars)
        )
        if ci_valid.any():
            lo = np.maximum(0.0, means[ci_valid] - lower_ci[ci_valid])
            hi = np.maximum(0.0, upper_ci[ci_valid] - means[ci_valid])
            ax.errorbar(
                positions[ci_valid], means[ci_valid],
                yerr=np.vstack([lo, hi]), fmt="none",
                ecolor=ERRORBAR_COLOR, capsize=4.5, capthick=1.7,
                elinewidth=1.7, zorder=5,
            )

        for j, (xpos, mean) in enumerate(zip(positions, means)):
            if not np.isfinite(mean):
                continue
            cap_top = upper_ci[j] if ci_valid[j] else mean
            drawn_limits.extend([float(mean), float(cap_top)])
            ax.annotate(
                f"{mean:,.2f}", xy=(xpos, cap_top), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color="#141414", clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.25),
                zorder=6,
            )

    labels = [SCENARIO_LABELS.get(name, name.replace("_", " ").title()) for name in scenarios]
    ax.set_xticks(x, labels)
    ax.set_xlabel("Climate/water scenario")
    ylabel, _ = METRIC_FORMAT.get(
        metric, (metric.replace("_", " ").title(), metric.replace("_", " ").title())
    )
    ax.set_ylabel(ylabel)
    # Do not add a graph title: figure captions belong in the report.

    if drawn_limits:
        ymin, ymax = ax.get_ylim()
        max_top = max(drawn_limits)
        span = max(max_top - ymin, 1.0)
        ax.set_ylim(ymin, max(ymax, max_top + 0.17 * span))
    ax.grid(axis="y", color="#D1D5DB", alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper left", frameon=False)
    fig.tight_layout()
    _save_svg(fig, outdir / FIGURE_FILENAMES.get(metric, f"{metric}_by_scenario.svg"))
    plt.close(fig)


def _scenario_design_outputs(root: Path, outdir: Path, summaries_dir: Path) -> None:
    records = []
    groups = []
    labels = []
    scenario_dirs = {p.name: p for p in (root / "scenarios").iterdir() if p.is_dir()}
    ordered_names = [name for name in SCENARIO_ORDER if name in scenario_dirs]
    ordered_names.extend(sorted(set(scenario_dirs).difference(ordered_names)))
    for scenario_name in ordered_names:
        scenario_dir = scenario_dirs[scenario_name]
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
        labels.append(SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title()))
    if not records:
        return
    pd.DataFrame(records).to_csv(summaries_dir / "scenario_year_split.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_ylabel("Seasonal precipitation (mm)")
    ax.set_xlabel("Scenario")
    # No top title on the rainfall figure either.
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save_svg(fig, outdir / "historical_seasonal_rainfall_by_scenario.svg")
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
    for path in sorted((output / "runs").glob("*/*/seed_*/interaction_budget.json")):
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
    baseline_seeds = [int(seed) for seed in plan.get("baseline_seeds", seeds[:1])]
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
                    except Exception as exc:
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
        ) if not frame.empty else 0,
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

    # Recalculate previously stored evaluation metrics using the original
    # paper's economics and water-efficiency formula. This keeps old raw run
    # files untouched and regenerates consistent derived report outputs.
    all_rows = _apply_baseline_formulas(
        collect_evaluations(root, benchmark["output_root"])
    )
    assumptions_path = output / "manifests" / "economic_assumptions.json"
    assumptions_path.parent.mkdir(parents=True, exist_ok=True)
    assumptions_path.write_text(
        json.dumps(
            {
                "source": "Alkaff et al., Mathematics 2025, 13(4), 595",
                "doi": "10.3390/math13040595",
                "crop_price_usd_per_tonne": BASELINE_CROP_PRICE_USD_PER_TONNE,
                "irrigation_cost_usd_per_ha_mm": BASELINE_IRRIGATION_COST_USD_PER_HA_MM,
                "fixed_non_irrigation_cost_usd_per_ha": BASELINE_FIXED_COST_USD_PER_HA,
                "profit_formula": "180 * yield_t_ha - 1.00 * irrigation_mm - 1728",
                "water_efficiency_formula": "1000 * yield_t_ha / irrigation_mm",
                "water_efficiency_when_irrigation_is_zero": "undefined (NaN)",
                "recomputed_in": "reporting pipeline before seed aggregation",
                "raw_run_evaluation_csv_files_modified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    primary = all_rows.sort_values("episode").drop_duplicates(
        ["algorithm", "scenario", "seed", "year"], keep="first"
    )
    all_rows.to_csv(evaluations_dir / "all_evaluation_rows.csv", index=False)
    primary.to_csv(evaluations_dir / "primary_seed_year_rows.csv", index=False)

    metrics = [metric for metric in METRIC_DIRECTIONS if metric in primary.columns]
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

    figure_index = []
    metrics_without_error_bars = {"water_productivity_kg_per_ha_per_mm", "profit"}
    for metric in metrics:
        _plot_metric(summary, metric, algorithms, figures_dir)
        figure_name = FIGURE_FILENAMES.get(metric, f"{metric}_by_scenario.svg")
        label, caption = METRIC_FORMAT.get(metric, (metric, metric))
        figure_index.append(
            {
                "metric": metric,
                "title": caption,
                "y_axis_label": label,
                "svg_file": figure_name,
                "scenario_order": "Extreme Dry; Dry; Normal; Limited Water; Wet",
                "error_bars": (
                    "None (baseline-style presentation)"
                    if metric in metrics_without_error_bars
                    else "95% percentile bootstrap CI across independent training seeds (n>=2)"
                ),
                "annotations": (
                    "Arithmetic mean above bar"
                    if metric in metrics_without_error_bars
                    else "Arithmetic mean above upper CI cap when CI is available"
                ),
            }
        )
    pd.DataFrame(figure_index).to_csv(summaries_dir / "figure_index.csv", index=False)

    return {
        "all_rows": evaluations_dir / "all_evaluation_rows.csv",
        "summary": summaries_dir / "algorithm_scenario_summary.csv",
        "friedman": statistics_dir / "friedman_tests.csv",
        "pairwise": statistics_dir / "pairwise_wilcoxon_holm.csv",
        "figures": figures_dir,
    }
