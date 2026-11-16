from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


METRIC_DIRECTIONS = {
    "dry_yield_tonne_per_ha": "max",
    "total_irrigation_mm": "min",
    "water_productivity_kg_per_ha_per_mm": "max",
    "profit": "max",
    "episode_reward": "max",
    "stress_exposure": "min",
    "stress_days": "min",
    "max_daily_stress": "min",
}


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float


def bootstrap_mean_ci(
    values: Iterable[float], *, resamples: int = 10000, alpha: float = 0.05, seed: int = 4279123
) -> BootstrapCI:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return BootstrapCI(np.nan, np.nan, np.nan)
    if array.size == 1:
        value = float(array[0])
        return BootstrapCI(value, value, value)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, array.size, size=(int(resamples), array.size))
    estimates = array[indexes].mean(axis=1)
    return BootstrapCI(
        float(array.mean()),
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_seeds(frame: pd.DataFrame, metric: str, algorithms: list[str]) -> pd.DataFrame:
    """Return one paired value per independent training seed.

    Policies trained with the same seed are evaluated over the same held-out years.
    Averaging those years before inference prevents the same trained policy from
    being counted repeatedly as if every seed-year row were independent. Climate-
    year detail remains available for descriptive and Pareto analyses.
    """

    clean = frame[frame["algorithm"].isin(algorithms)].copy()
    clean = clean.groupby(["scenario", "seed", "algorithm"], as_index=False)[metric].mean()
    pivot = clean.pivot(index="seed", columns="algorithm", values=metric)
    return pivot.reindex(columns=algorithms).dropna()


def friedman_table(
    frame: pd.DataFrame, metrics: list[str], algorithms: list[str]
) -> pd.DataFrame:
    rows = []
    for scenario, scenario_frame in frame.groupby("scenario"):
        for metric in metrics:
            pivot = paired_seeds(scenario_frame, metric, algorithms)
            if len(pivot) < 2 or len(algorithms) < 3:
                statistic = p_value = np.nan
            else:
                result = stats.friedmanchisquare(*(pivot[name].to_numpy() for name in algorithms))
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n_paired_seeds": len(pivot),
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm_across_tests"] = holm_adjust(result["p_value"].fillna(1.0).tolist())
    return result


def pairwise_wilcoxon_table(
    frame: pd.DataFrame, metrics: list[str], algorithms: list[str]
) -> pd.DataFrame:
    rows = []
    for scenario, scenario_frame in frame.groupby("scenario"):
        for metric in metrics:
            metric_rows = []
            for algorithm_a, algorithm_b in itertools.combinations(algorithms, 2):
                pivot = paired_seeds(scenario_frame, metric, [algorithm_a, algorithm_b])
                differences = pivot[algorithm_a] - pivot[algorithm_b]
                nonzero = differences[np.abs(differences) > 1e-12]
                if len(nonzero) == 0:
                    statistic, p_value = 0.0, 1.0
                else:
                    result = stats.wilcoxon(
                        pivot[algorithm_a],
                        pivot[algorithm_b],
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                direction = METRIC_DIRECTIONS.get(metric, "max")
                signed = differences if direction == "max" else -differences
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "algorithm_a": algorithm_a,
                        "algorithm_b": algorithm_b,
                        "n_paired_seeds": len(pivot),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "median_raw_difference_a_minus_b": float(differences.median()),
                        "a_win_rate": float((signed > 0).mean()),
                        "ties_rate": float((np.abs(signed) <= 1e-12).mean()),
                    }
                )
            adjusted = holm_adjust([row["p_value"] for row in metric_rows])
            for row, p_adjusted in zip(metric_rows, adjusted):
                row["p_holm_within_scenario_metric"] = p_adjusted
                rows.append(row)
    return pd.DataFrame(rows)


def pareto_frequency(frame: pd.DataFrame, algorithms: list[str]) -> pd.DataFrame:
    metrics = ["dry_yield_tonne_per_ha", "total_irrigation_mm"]
    grouped = frame[frame["algorithm"].isin(algorithms)].groupby(
        ["scenario", "seed", "year", "algorithm"], as_index=False
    )[metrics].mean()
    rows = []
    for (scenario, seed, year), block in grouped.groupby(["scenario", "seed", "year"]):
        block = block.set_index("algorithm")
        for algorithm in algorithms:
            if algorithm not in block.index:
                continue
            y = block.loc[algorithm, "dry_yield_tonne_per_ha"]
            w = block.loc[algorithm, "total_irrigation_mm"]
            dominated = False
            for competitor in algorithms:
                if competitor == algorithm or competitor not in block.index:
                    continue
                cy = block.loc[competitor, "dry_yield_tonne_per_ha"]
                cw = block.loc[competitor, "total_irrigation_mm"]
                if cy >= y and cw <= w and (cy > y or cw < w):
                    dominated = True
                    break
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "year": year,
                    "algorithm": algorithm,
                    "pareto_efficient": not dominated,
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    return (
        detail.groupby(["scenario", "algorithm"], as_index=False)
        .agg(pareto_frequency=("pareto_efficient", "mean"), blocks=("pareto_efficient", "size"))
    )
