"""Reference AquaCrop conventional strategies (independent of RL action gate).

Alkaff et al. (2025) used native AquaCrop management methods when comparing
PPO against conventional irrigation. These policies intentionally do not call
the learned agent's 40%-depletion action transform. This is a COMPARATOR
implementation; continuous RL actions, 40% trigger, 25 mm limit are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from maize_benchmark.config import load_benchmark_config
from maize_benchmark.envs.scenarios import load_scenario
from maize_benchmark.evaluation import FINAL_COLUMNS, save_evaluation
from maize_benchmark.protocol import evaluation_metrics


def native_irrigation_settings(baseline: str, cap_mm: float | None) -> dict:
    """Map documented paper strategies to AquaCrop IrrigationManagement args."""
    common = {"MaxIrr": 25.0}
    if cap_mm is not None:
        if not np.isfinite(cap_mm) or cap_mm < 0:
            raise ValueError("Seasonal irrigation cap must be finite and non-negative")
        common["MaxIrrSeason"] = float(cap_mm)
    if baseline == "rainfed":
        return {"irrigation_method": 0, **common}
    if baseline == "soil_moisture_threshold":
        return {
            "irrigation_method": 1,
            "SMT": [23.72, 26.46, 38.19, 50.11],
            **common,
        }
    if baseline == "fixed_interval":
        return {"irrigation_method": 2, "IrrInterval": 7, **common}
    raise ValueError(f"Unknown reference irrigation strategy {baseline!r}")


def evaluate_baseline(
    *,
    project_root: str | Path,
    baseline: str,
    scenario: str,
    seed: int,
    episodes_per_year: int,
    split: str = "test",
) -> Path:
    """Evaluate once per held-out climate year; never fake seed replication."""
    if int(episodes_per_year) != 1:
        raise ValueError(
            "Native deterministic conventional baselines use exactly one "
            "simulation per held-out year. Repeated runs of the same weather "
            "are not independent 100-episode samples."
        )
    root = Path(project_root).resolve()
    benchmark = load_benchmark_config(root)
    scenario_obj = load_scenario(root, scenario)
    if split == "test":
        years = scenario_obj.test_years
    elif split == "validation":
        years = scenario_obj.validation_years
    else:
        raise ValueError(f"Unknown evaluation split: {split}")

    # Imports are deferred so pure reporting and unit tests do not require
    # AquaCrop's optional compiled simulation dependencies.
    from aquacrop.core import AquaCropModel
    from aquacrop.entities.crop import Crop
    from aquacrop.entities.inititalWaterContent import InitialWaterContent
    from aquacrop.entities.irrigationManagement import IrrigationManagement
    from aquacrop.entities.soil import Soil
    from aquacrop.utils import prepare_weather

    weather = prepare_weather(str(root / benchmark["climate_file"]))
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather["Year"] = weather["Date"].dt.year
    rows = []
    for year in years:
        year_weather = weather.loc[weather["Year"] == int(year)].copy()
        if year_weather.empty:
            raise ValueError(f"No climate records for {year}")
        cap = scenario_obj.cap_for_year(int(year))
        management = IrrigationManagement(
            **native_irrigation_settings(baseline, cap)
        )
        model = AquaCropModel(
            f"{year}/{benchmark['planting_date']}",
            f"{year}/{benchmark['season_end']}",
            year_weather.reset_index(drop=True),
            Soil(benchmark["soil"]),
            Crop(benchmark["crop"], planting_date=benchmark["planting_date"]),
            irrigation_management=management,
            initial_water_content=InitialWaterContent(value=["FC"]),
        )
        model.run_model(till_termination=True)
        stats = model._outputs.final_stats
        if stats is None or len(stats) == 0:
            raise RuntimeError(f"No native AquaCrop final results for {baseline} {scenario} {year}")
        yield_t_ha = float(stats["Dry yield (tonne/ha)"].mean())
        irrigation_mm = float(stats["Seasonal irrigation (mm)"].mean())
        if cap is not None and irrigation_mm > cap + 1e-4:
            raise RuntimeError(
                f"AquaCrop baseline {baseline} overran limited-water allocation "
                f"in {year}: applied {irrigation_mm:.3f} mm; cap {cap:.3f} mm"
            )
        metrics = evaluation_metrics(
            dry_yield_tonne_per_ha=yield_t_ha,
            seasonal_irrigation_mm=irrigation_mm,
        )
        row = {key: np.nan for key in FINAL_COLUMNS}
        row.update({
            "algorithm": baseline, "scenario": scenario, "seed": int(seed),
            "year": int(year), "episode": 0,
            "dry_yield_tonne_per_ha": yield_t_ha,
            "total_irrigation_mm": irrigation_mm,
            "model_seasonal_irrigation_mm": irrigation_mm,
            "irrigation_cap_mm": cap,
            "water_productivity_kg_per_ha_per_mm": metrics["water_productivity_kg_per_ha_per_mm"],
            "profit": metrics["profit"],
        })
        rows.append(row)

    target = (
        root / benchmark["output_root"] / "runs" / baseline / scenario
        / f"seed_{seed}" / ("evaluation.csv" if split == "test" else "validation_evaluation.csv")
    )
    save_evaluation(pd.DataFrame(rows, columns=FINAL_COLUMNS), target)
    return target
