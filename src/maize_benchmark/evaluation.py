from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from maize_benchmark.envs.factory import build_env


class StatefulPolicy(Protocol):
    def __call__(self, obs: np.ndarray, deterministic: bool) -> np.ndarray: ...

    def reset_episode(self) -> None: ...


PolicyFn = Callable[[np.ndarray, bool], np.ndarray] | StatefulPolicy


FINAL_COLUMNS = [
    "algorithm",
    "scenario",
    "seed",
    "year",
    "episode",
    "dry_yield_tonne_per_ha",
    "total_irrigation_mm",
    "water_productivity_kg_per_ha_per_mm",
    "profit",
    "episode_reward",
    "stress_exposure",
    "stress_days",
    "max_daily_stress",
    "days",
    "irrigation_cap_mm",
    "model_seasonal_irrigation_mm",
]


def evaluate_policy(
    *,
    project_root: str | Path,
    algorithm: str,
    scenario: str,
    seed: int,
    policy: PolicyFn,
    episodes_per_year: int,
    deterministic: bool = True,
    split: str = "test",
) -> pd.DataFrame:
    """Evaluate on fixed years without silently auto-resetting environments.

    Deterministic evaluation uses one rollout per seed-year. More than one rollout
    is permitted for explicitly stochastic robustness analyses. Primary inferential
    reporting averages held-out years within each independently trained seed.
    """

    root = Path(project_root)
    from maize_benchmark.envs.scenarios import load_scenario

    scenario_obj = load_scenario(root, scenario)
    if split == "validation":
        years = scenario_obj.validation_years
    elif split == "test":
        years = scenario_obj.test_years
    else:
        raise ValueError("Evaluation split must be 'validation' or 'test'")

    rows: list[dict[str, Any]] = []
    for year in years:
        for episode in range(int(episodes_per_year)):
            eval_seed = int(seed * 100000 + year * 100 + episode)
            env = build_env(root, scenario, split=split, seed=eval_seed, fixed_year=year)
            obs, _ = env.reset(seed=eval_seed, options={"year": year})
            reset_policy = getattr(policy, "reset_episode", None)
            if callable(reset_policy):
                reset_policy()
            terminated = truncated = False
            final_info: dict[str, Any] | None = None
            while not (terminated or truncated):
                action = policy(np.asarray(obs, dtype=np.float32), deterministic)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    final_info = info
            env.close()
            if not final_info:
                raise RuntimeError("Evaluation episode ended without final metrics")
            rows.append(
                {
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "seed": int(seed),
                    "year": int(year),
                    "episode": int(episode),
                    **{key: final_info.get(key) for key in FINAL_COLUMNS[5:]},
                }
            )
    frame = pd.DataFrame(rows)
    for column in FINAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[FINAL_COLUMNS]


def save_evaluation(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
