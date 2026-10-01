import numpy as np
import pandas as pd

from maize_benchmark.reporting.statistics import (
    bootstrap_mean_ci,
    friedman_table,
    holm_adjust,
    pairwise_wilcoxon_table,
    pareto_frequency,
)


def synthetic_frame():
    rows = []
    algorithms = ["ppo", "crossq", "dsac_t", "dreamerv3"]
    for seed in [11, 23, 37]:
        for year in [2000, 2001]:
            for index, algorithm in enumerate(algorithms):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "scenario": "dry",
                        "seed": seed,
                        "year": year,
                        "dry_yield_tonne_per_ha": 10 + index,
                        "total_irrigation_mm": 500 + index * 10,
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_ci_is_ordered():
    result = bootstrap_mean_ci([1, 2, 3, 4], resamples=500, seed=1)
    assert result.lower <= result.mean <= result.upper


def test_holm_is_monotone_in_sorted_order():
    raw = [0.04, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    sorted_adjusted = np.asarray(adjusted)[order]
    assert np.all(np.diff(sorted_adjusted) >= -1e-12)
    assert all(0 <= value <= 1 for value in adjusted)


def test_paired_statistics_and_pareto():
    frame = synthetic_frame()
    algorithms = ["ppo", "crossq", "dsac_t", "dreamerv3"]
    friedman = friedman_table(frame, ["dry_yield_tonne_per_ha"], algorithms)
    assert friedman.loc[0, "n_paired_seeds"] == 3
    pairwise = pairwise_wilcoxon_table(frame, ["dry_yield_tonne_per_ha"], algorithms)
    assert len(pairwise) == 6
    pareto = pareto_frequency(frame, algorithms)
    assert set(pareto["algorithm"]) == set(algorithms)
