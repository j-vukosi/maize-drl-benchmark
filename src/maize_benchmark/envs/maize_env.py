from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("DEVELOPMENT", "DEVELOPMENT")

OBSERVATION_SIZE = 26
DEPLETION_TRIGGER_FRACTION = 0.40


class AquaCropMaizeBenchmarkEnv:
    """Gymnasium-compatible AquaCrop maize environment with benchmark controls.

    Imports are delayed so scenario preparation and reporting can run without
    AquaCrop. The environment is constructed through ``make_maize_env``, which
    dynamically subclasses Gymnasium's Env type.
    """


def make_maize_env(
    *,
    climate_file: str | Path,
    years: Sequence[int],
    seed: int,
    crop_name: str = "Maize",
    soil_name: str = "SandyLoam",
    planting_date: str = "05/01",
    season_end: str = "12/31",
    max_daily_irrigation_mm: float = 25.0,
    min_irrigation_event_mm: float = 0.0,
    irrigation_cap_mm: float | None = None,
    cap_lookup: dict[int, float] | None = None,
    action_mode: str = "continuous_depth",
    reward_version: str = "balanced_v2",
    reward_config: dict[str, float] | None = None,
    profit_config: dict[str, float] | None = None,
    fixed_year: int | None = None,
):
    try:
        import gymnasium as gym
        from aquacrop.core import AquaCropModel
        from aquacrop.entities.crop import Crop
        from aquacrop.entities.inititalWaterContent import InitialWaterContent
        from aquacrop.entities.irrigationManagement import IrrigationManagement
        from aquacrop.entities.soil import Soil
        from aquacrop.utils import prepare_weather
        from gymnasium import spaces
    except ImportError as exc:
        raise RuntimeError(
            "AquaCropGymnasium dependencies are unavailable. "
            "Run scripts/bootstrap_windows.ps1."
        ) from exc

    years_tuple = tuple(sorted({int(year) for year in years}))

    if not years_tuple:
        raise ValueError("At least one scenario year is required.")

    if fixed_year is not None and int(fixed_year) not in years_tuple:
        raise ValueError(
            f"Fixed year {fixed_year} is not included in permitted years "
            f"{years_tuple}."
        )

    if max_daily_irrigation_mm <= 0:
        raise ValueError("max_daily_irrigation_mm must be greater than zero.")

    if irrigation_cap_mm is not None and irrigation_cap_mm < 0:
        raise ValueError("irrigation_cap_mm cannot be negative.")

    reward_cfg = {
        "yield_power": 4.0,
        "yield_scale": 1.0,
        "water_penalty_per_mm": 1.0,
        "stress_penalty_per_day": 0.0,
        **(reward_config or {}),
    }

    profit_cfg = {
        "maize_price_per_tonne": 220.0,
        "irrigation_cost_per_mm_ha": 1.10,
        "fixed_cost_per_ha": 0.0,
        **(profit_config or {}),
    }

    resolved_cap_lookup = {
        int(year): float(cap)
        for year, cap in (cap_lookup or {}).items()
    }

    for year, cap in resolved_cap_lookup.items():
        if cap < 0:
            raise ValueError(
                f"Irrigation cap for year {year} cannot be negative: {cap}"
            )

    class _MaizeEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()

            self.climate_file = Path(climate_file).resolve()

            if not self.climate_file.is_file():
                raise FileNotFoundError(
                    f"Climate file does not exist: {self.climate_file}"
                )

            self.years = years_tuple
            self.base_seed = int(seed)
            self.crop_name = str(crop_name)
            self.soil_name = str(soil_name)
            self.planting_date = str(planting_date)
            self.season_end = str(season_end)
            self.max_daily_irrigation_mm = float(max_daily_irrigation_mm)
            self.min_irrigation_event_mm = float(min_irrigation_event_mm)

            if (
                not np.isfinite(self.min_irrigation_event_mm)
                or self.min_irrigation_event_mm < 0.0
                or self.min_irrigation_event_mm > self.max_daily_irrigation_mm
            ):
                raise ValueError(
                    "min_irrigation_event_mm must be finite and within "
                    f"[0, {self.max_daily_irrigation_mm}], received "
                    f"{self.min_irrigation_event_mm}."
                )
            self.irrigation_cap_mm = (
                None
                if irrigation_cap_mm is None
                else float(irrigation_cap_mm)
            )
            self.action_mode = str(action_mode)
            self.reward_version = str(reward_version)
            self.fixed_year = (
                None if fixed_year is None else int(fixed_year)
            )
            self.cap_lookup = resolved_cap_lookup.copy()

            # Explicit float32 arrays prevent Gymnasium from first creating
            # float64 bounds and then warning that precision was lowered.
            action_low = np.array([0.0], dtype=np.float32)
            action_high = np.array(
                [self.max_daily_irrigation_mm],
                dtype=np.float32,
            )

            self.action_space = spaces.Box(
                low=action_low,
                high=action_high,
                dtype=np.float32,
            )

            observation_low = np.full(
                (OBSERVATION_SIZE,),
                -np.inf,
                dtype=np.float32,
            )
            observation_high = np.full(
                (OBSERVATION_SIZE,),
                np.inf,
                dtype=np.float32,
            )

            self.observation_space = spaces.Box(
                low=observation_low,
                high=observation_high,
                dtype=np.float32,
            )

            self._all_weather = prepare_weather(str(self.climate_file))

            # AquaCrop versions differ on whether prepare_weather retains
            # a Year column. Derive it from Date when necessary.
            if "Year" not in self._all_weather.columns:
                if "Date" not in self._all_weather.columns:
                    raise ValueError(
                        "Prepared weather data must contain either "
                        "'Year' or 'Date'."
                    )

                import pandas as pd

                self._all_weather["Year"] = pd.to_datetime(
                    self._all_weather["Date"],
                    errors="raise",
                ).dt.year

            self._all_weather["Year"] = (
                self._all_weather["Year"].astype(int)
            )

            available_years = set(
                self._all_weather["Year"].unique().tolist()
            )
            missing_years = sorted(set(self.years) - available_years)

            if missing_years:
                raise ValueError(
                    "Climate data does not contain all requested years. "
                    f"Missing: {missing_years}"
                )

            self._initial_wc = InitialWaterContent(value=["FC"])
            self._soil = Soil(self.soil_name)

            self.model = None
            self.weather = None
            self.year: int | None = None

            self.total_irrigation_mm = 0.0
            self.cumulative_reward = 0.0
            self.stress_exposure = 0.0
            self.stress_days = 0
            self.max_daily_stress = 0.0
            self.days = 0

            # Shared water-demand diagnostics for every algorithm.
            self.total_rainfall_mm = 0.0
            self.irrigation_event_count = 0
            self.sum_root_zone_depletion_before_mm = 0.0
            self.max_root_zone_depletion_before_mm = 0.0
            self.last_root_zone_depletion_before_mm = 0.0
            self.last_root_zone_taw_before_mm = 0.0
            self.last_irrigation_action_fraction = 0.0
            self.last_requested_irrigation_mm = 0.0
            self.last_applied_irrigation_mm = 0.0
            self.last_rainfall_mm = 0.0
            self.water_balance_trace: list[dict[str, float | int]] = []

            self._has_reset = False

            self.action_space.seed(self.base_seed)

        def _select_year(
            self,
            options: dict[str, Any] | None,
        ) -> int:
            requested = None if options is None else options.get("year")

            if requested is not None:
                requested = int(requested)

                if requested not in self.years:
                    raise ValueError(
                        f"Year {requested} not permitted: {self.years}"
                    )

                return requested

            if self.fixed_year is not None:
                return self.fixed_year

            return int(self.np_random.choice(self.years))

        def _validate_observation(
            self,
            observation: Any,
        ) -> np.ndarray:
            """Return a contiguous float32 observation matching the space."""

            obs = np.asarray(observation, dtype=np.float32)

            if obs.shape != self.observation_space.shape:
                raise ValueError(
                    "Observation shape mismatch: "
                    f"received {obs.shape}, "
                    f"expected {self.observation_space.shape}."
                )

            if not np.all(np.isfinite(obs)):
                invalid_indices = np.flatnonzero(~np.isfinite(obs))

                raise ValueError(
                    "Observation contains non-finite values at indices "
                    f"{invalid_indices.tolist()}: {obs}"
                )

            obs = np.ascontiguousarray(obs, dtype=np.float32)

            if not self.observation_space.contains(obs):
                below_indices = np.flatnonzero(
                    obs < self.observation_space.low
                )
                above_indices = np.flatnonzero(
                    obs > self.observation_space.high
                )

                raise ValueError(
                    "Observation is outside the declared observation space.\n"
                    f"Observation: {obs}\n"
                    f"Observation dtype: {obs.dtype}\n"
                    f"Expected dtype: {self.observation_space.dtype}\n"
                    f"Lower bounds: {self.observation_space.low}\n"
                    f"Upper bounds: {self.observation_space.high}\n"
                    f"Below-bound indices: {below_indices.tolist()}\n"
                    f"Above-bound indices: {above_indices.tolist()}"
                )

            return obs

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ):
            effective_seed = seed

            if not self._has_reset and effective_seed is None:
                effective_seed = self.base_seed

            super().reset(seed=effective_seed)

            if effective_seed is not None:
                self.action_space.seed(int(effective_seed))

            self._has_reset = True
            self.year = self._select_year(options)

            year_weather = self._all_weather[
                self._all_weather["Year"] == self.year
            ].copy()

            if year_weather.empty:
                raise ValueError(
                    f"No weather rows found for year {self.year}."
                )

            import pandas as pd

            year_weather["Date"] = pd.to_datetime(
                year_weather["Date"],
                errors="raise",
            )

            simulation_start = pd.Timestamp(
                f"{self.year}/{self.planting_date}"
            )
            simulation_end = pd.Timestamp(
                f"{self.year}/{self.season_end}"
            )

            # The AquaCrop clock counter is relative to the simulation
            # start, not to January 1. Keep a crop-season weather frame
            # for the RL observation/diagnostic layer so counter 0/1 maps
            # to May 1/May 2 rather than Jan 1/Jan 2.
            self.weather = year_weather.loc[
                (year_weather["Date"] >= simulation_start)
                & (year_weather["Date"] <= simulation_end)
            ].reset_index(drop=True)

            if self.weather.empty:
                raise ValueError(
                    "No weather rows fall inside the simulation window "
                    f"{simulation_start.date()} to {simulation_end.date()}."
                )

            crop = Crop(
                self.crop_name,
                self.planting_date,
            )

            self.model = AquaCropModel(
                f"{self.year}/{self.planting_date}",
                f"{self.year}/{self.season_end}",
                year_weather.reset_index(drop=True),
                self._soil,
                crop,
                irrigation_management=IrrigationManagement(
                    irrigation_method=5
                ),
                initial_water_content=self._initial_wc,
            )

            self.model.run_model()

            self.total_irrigation_mm = 0.0
            self.cumulative_reward = 0.0
            self.stress_exposure = 0.0
            self.stress_days = 0
            self.max_daily_stress = 0.0
            self.days = 0

            self.total_rainfall_mm = 0.0
            self.irrigation_event_count = 0
            self.sum_root_zone_depletion_before_mm = 0.0
            self.max_root_zone_depletion_before_mm = 0.0
            self.last_root_zone_depletion_before_mm = 0.0
            self.last_root_zone_taw_before_mm = 0.0
            self.last_irrigation_action_fraction = 0.0
            self.last_requested_irrigation_mm = 0.0
            self.last_applied_irrigation_mm = 0.0
            self.last_rainfall_mm = 0.0
            self.water_balance_trace = []

            observation = self._validate_observation(self._get_obs())
            info = self._info(final=False)

            return observation, info

        def _get_last_values(
            self,
            column: str,
            count: int = 7,
        ) -> np.ndarray:
            if self.model is None or self.weather is None:
                raise RuntimeError(
                    "Environment must be reset before observations "
                    "can be generated."
                )

            if column not in self.weather.columns:
                raise KeyError(
                    f"Weather column is missing: {column}"
                )

            current = max(
                0,
                int(self.model._clock_struct.time_step_counter),
            )

            start = max(0, current - count)

            values = self.weather.iloc[start:current][column].to_numpy(
                dtype=np.float32
            )

            if values.size < count:
                values = np.pad(
                    values,
                    (count - values.size, 0),
                    mode="constant",
                    constant_values=0.0,
                )

            return np.asarray(values, dtype=np.float32)

        def _get_obs(self) -> np.ndarray:
            if self.model is None:
                raise RuntimeError(
                    "Environment must be reset before an observation "
                    "can be generated."
                )

            condition = self.model._init_cond

            weather_features = np.concatenate(
                [
                    self._get_last_values("Precipitation"),
                    self._get_last_values("MinTemp"),
                    self._get_last_values("MaxTemp"),
                ]
            ).astype(np.float32, copy=False)

            crop_features = np.array(
                [
                    condition.age_days,
                    condition.canopy_cover,
                    condition.biomass,
                    condition.depletion,
                    condition.taw,
                ],
                dtype=np.float32,
            )

            observation = np.concatenate(
                [crop_features, weather_features]
            )

            return np.asarray(observation, dtype=np.float32)

        def _year_cap(self) -> float | None:
            if self.year is not None and self.year in self.cap_lookup:
                return float(self.cap_lookup[self.year])

            return self.irrigation_cap_mm

        def _root_zone_water_status(self) -> tuple[float, float]:
            """Return current root-zone depletion and total available water."""
            if self.model is None:
                raise RuntimeError(
                    "Environment must be reset before water status "
                    "can be calculated."
                )

            condition = self.model._init_cond
            depletion = max(0.0, float(condition.depletion))
            taw = max(0.0, float(condition.taw))

            if not np.isfinite(depletion) or not np.isfinite(taw):
                raise ValueError(
                    "AquaCrop returned non-finite root-zone water status: "
                    f"depletion={depletion}, taw={taw}."
                )

            return depletion, taw

        def _current_weather_value(self, column: str) -> float:
            """Return the weather value for the day about to be simulated."""
            if self.model is None or self.weather is None:
                raise RuntimeError(
                    "Environment must be reset before weather can be read."
                )

            if column not in self.weather.columns:
                raise KeyError(f"Weather column is missing: {column}")

            current = int(self.model._clock_struct.time_step_counter)

            if current < 0 or current >= len(self.weather):
                return 0.0

            value = float(self.weather.iloc[current][column])

            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite weather value for {column}: {value}"
                )

            return value

        def _map_action(self, action: Any) -> float:
            action_array = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

            if action_array.size != 1:
                raise ValueError(
                    "Expected one irrigation action value, "
                    f"received shape {np.asarray(action).shape}."
                )

            raw = float(action_array[0])

            if not np.isfinite(raw):
                raise ValueError(
                    f"Action must be finite, received {raw}."
                )

            raw = float(
                np.clip(
                    raw,
                    0.0,
                    self.max_daily_irrigation_mm,
                )
            )

            root_zone_depletion_mm, root_zone_taw_mm = (
                self._root_zone_water_status()
            )

            # The existing normalized wrapper maps [-1, 1] into the native
            # [0, max_daily_irrigation_mm] range. Dividing here therefore
            # recovers a common [0, 1] requested fraction.
            action_fraction = (
                raw / self.max_daily_irrigation_mm
                if self.max_daily_irrigation_mm > 0.0
                else 0.0
            )

            if self.action_mode == "binary_ablation":
                threshold = self.max_daily_irrigation_mm / 2.0
                requested = (
                    self.max_daily_irrigation_mm
                    if raw >= threshold
                    else 0.0
                )
                action_fraction = 1.0 if requested > 0.0 else 0.0
            elif self.action_mode == "continuous_depth":
                # Preserve the original formulation for reproducibility.
                requested = raw
            elif self.action_mode == "deficit_fraction":
                # Revised formulation: irrigation is physically enabled only
                # once root-zone depletion reaches the frozen 40% TAW trigger.
                # Above the trigger, the policy chooses what fraction of the
                # CURRENT root-zone deficit to refill.
                depletion_fraction = (
                    root_zone_depletion_mm / root_zone_taw_mm
                    if root_zone_taw_mm > 0.0
                    else 0.0
                )

                if (
                    depletion_fraction
                    < DEPLETION_TRIGGER_FRACTION
                ):
                    requested = 0.0
                else:
                    requested = (
                        action_fraction * root_zone_depletion_mm
                    )
            else:
                raise ValueError(
                    f"Unknown action_mode={self.action_mode}"
                )

            requested = max(0.0, float(requested))

            applied = min(
                requested,
                self.max_daily_irrigation_mm,
            )

            if self.action_mode == "deficit_fraction":
                applied = min(
                    applied,
                    root_zone_depletion_mm,
                )

            cap = self._year_cap()

            if cap is not None:
                remaining = max(
                    0.0,
                    float(cap) - self.total_irrigation_mm,
                )
                applied = min(applied, remaining)

            # Retain the existing 1 mm deadband without allowing it to push an
            # irrigation event above the current deficit or seasonal cap.
            tolerance = 1e-6

            if (
                applied > 0.0
                and applied
                < self.min_irrigation_event_mm - tolerance
            ):
                applied = 0.0
            elif (
                applied > 0.0
                and self.min_irrigation_event_mm > 0.0
            ):
                constraint_limit = min(
                    self.max_daily_irrigation_mm,
                    (
                        root_zone_depletion_mm
                        if self.action_mode == "deficit_fraction"
                        else self.max_daily_irrigation_mm
                    ),
                    (
                        max(
                            0.0,
                            float(cap) - self.total_irrigation_mm,
                        )
                        if cap is not None
                        else self.max_daily_irrigation_mm
                    ),
                )

                if (
                    constraint_limit
                    < self.min_irrigation_event_mm - tolerance
                ):
                    applied = 0.0
                elif (
                    applied
                    >= self.min_irrigation_event_mm - tolerance
                ):
                    applied = min(
                        max(
                            applied,
                            self.min_irrigation_event_mm,
                        ),
                        constraint_limit,
                    )

            self.last_root_zone_depletion_before_mm = float(
                root_zone_depletion_mm
            )
            self.last_root_zone_taw_before_mm = float(root_zone_taw_mm)
            self.last_irrigation_action_fraction = float(action_fraction)
            self.last_requested_irrigation_mm = float(requested)
            self.last_applied_irrigation_mm = float(applied)

            return float(applied)

        def _reward(

            self,

            applied_mm: float,

            terminated: bool,

            dry_yield: float,

        ) -> float:

            if self.reward_version == "legacy_source":

                step_reward = (

                    -self.total_irrigation_mm

                    if applied_mm > 0

                    else 0.0

                )



                if terminated:

                    step_reward += (

                        reward_cfg["yield_scale"]

                        * dry_yield ** reward_cfg["yield_power"]

                    )



            elif self.reward_version == "balanced_v2":

                step_reward = (

                    -reward_cfg["water_penalty_per_mm"]

                    * applied_mm

                )

                step_reward -= (

                    reward_cfg["stress_penalty_per_day"]

                    * self._daily_stress()

                )



                if terminated:

                    step_reward += (

                        reward_cfg["yield_scale"]

                        * dry_yield ** reward_cfg["yield_power"]

                    )



            elif self.reward_version == "economic_v1":

                # Dense irrigation cost with terminal crop value.

                # The full episode return therefore equals the

                # benchmark profit metric:

                #

                # maize_price * seasonal_yield

                # - irrigation_cost * seasonal_irrigation

                # - fixed_cost

                step_reward = (

                    -profit_cfg["irrigation_cost_per_mm_ha"]

                    * applied_mm

                )



                if terminated:

                    step_reward += (

                        profit_cfg["maize_price_per_tonne"]

                        * dry_yield

                        - profit_cfg["fixed_cost_per_ha"]

                    )



            else:

                raise ValueError(

                    f"Unknown reward_version={self.reward_version}"

                )



            return float(step_reward)



        def _daily_stress(self) -> float:
            if self.model is None:
                raise RuntimeError(
                    "Environment must be reset before stress "
                    "can be calculated."
                )

            condition = self.model._init_cond
            taw = max(float(condition.taw), 1e-6)

            return float(
                np.clip(
                    float(condition.depletion) / taw,
                    0.0,
                    2.0,
                )
            )

        def _final_stats(self) -> tuple[float, float]:
            if self.model is None:
                raise RuntimeError(
                    "Environment must be reset before final statistics "
                    "can be calculated."
                )

            stats = self.model._outputs.final_stats

            if stats is None or len(stats) == 0:
                raise RuntimeError(
                    "AquaCrop produced no final season statistics."
                )

            dry_yield = float(
                stats["Dry yield (tonne/ha)"].mean()
            )
            irrigation = float(
                stats["Seasonal irrigation (mm)"].mean()
            )

            return dry_yield, irrigation

        def _info(
            self,
            final: bool,
            dry_yield: float = 0.0,
        ) -> dict[str, Any]:
            irrigation = float(self.total_irrigation_mm)

            water_productivity = (
                dry_yield * 1000.0 / irrigation
                if irrigation > 0
                else np.nan
            )

            profit = (
                dry_yield
                * profit_cfg["maize_price_per_tonne"]
                - irrigation
                * profit_cfg["irrigation_cost_per_mm_ha"]
                - profit_cfg["fixed_cost_per_ha"]
            )

            return {
                "year": (
                    int(self.year)
                    if self.year is not None
                    else None
                ),
                "dry_yield_tonne_per_ha": float(dry_yield),
                "total_irrigation_mm": irrigation,
                "water_productivity_kg_per_ha_per_mm": float(
                    water_productivity
                ),
                "profit": float(profit),
                "episode_reward": float(self.cumulative_reward),
                "stress_exposure": float(self.stress_exposure),
                "stress_days": int(self.stress_days),
                "max_daily_stress": float(self.max_daily_stress),
                "days": int(self.days),
                "final": bool(final),
                "irrigation_cap_mm": self._year_cap(),
                "seasonal_rainfall_to_date_mm": float(
                    self.total_rainfall_mm
                ),
                "irrigation_event_count": int(
                    self.irrigation_event_count
                ),
                "mean_root_zone_depletion_before_mm": float(
                    self.sum_root_zone_depletion_before_mm / self.days
                    if self.days > 0
                    else 0.0
                ),
                "max_root_zone_depletion_before_mm": float(
                    self.max_root_zone_depletion_before_mm
                ),
                "last_root_zone_depletion_before_mm": float(
                    self.last_root_zone_depletion_before_mm
                ),
                "last_root_zone_taw_before_mm": float(
                    self.last_root_zone_taw_before_mm
                ),
                "depletion_trigger_fraction": float(
                    DEPLETION_TRIGGER_FRACTION
                ),
                "last_root_zone_depletion_fraction": float(
                    self.last_root_zone_depletion_before_mm
                    / self.last_root_zone_taw_before_mm
                    if self.last_root_zone_taw_before_mm > 0.0
                    else 0.0
                ),
                "last_irrigation_action_fraction": float(
                    self.last_irrigation_action_fraction
                ),
                "last_requested_irrigation_mm": float(
                    self.last_requested_irrigation_mm
                ),
                "last_applied_irrigation_mm": float(
                    self.last_applied_irrigation_mm
                ),
                "last_rainfall_mm": float(self.last_rainfall_mm),
            }

        def step(self, action: Any):
            if self.model is None or not self._has_reset:
                raise RuntimeError(
                    "reset() must be called before step()."
                )

            if bool(self.model._clock_struct.model_is_finished):
                raise RuntimeError(
                    "step() was called after the episode terminated. "
                    "Call reset() before starting another episode."
                )

            rainfall_mm = max(
                0.0,
                self._current_weather_value("Precipitation"),
            )

            depth = self._map_action(action)

            root_zone_depletion_before_mm = float(
                self.last_root_zone_depletion_before_mm
            )
            root_zone_taw_before_mm = float(
                self.last_root_zone_taw_before_mm
            )
            action_fraction = float(
                self.last_irrigation_action_fraction
            )
            requested_irrigation_mm = float(
                self.last_requested_irrigation_mm
            )

            self.model._param_struct.IrrMngt.depth = depth
            self.model.run_model(initialize_model=False)

            self.total_irrigation_mm += depth
            self.total_rainfall_mm += rainfall_mm
            self.irrigation_event_count += int(depth > 0.0)
            self.sum_root_zone_depletion_before_mm += (
                root_zone_depletion_before_mm
            )
            self.max_root_zone_depletion_before_mm = max(
                self.max_root_zone_depletion_before_mm,
                root_zone_depletion_before_mm,
            )
            self.last_rainfall_mm = float(rainfall_mm)
            self.last_applied_irrigation_mm = float(depth)
            self.days += 1

            daily_stress = self._daily_stress()
            self.stress_exposure += daily_stress
            self.stress_days += int(daily_stress >= 0.5)
            self.max_daily_stress = max(
                self.max_daily_stress,
                daily_stress,
            )

            root_zone_depletion_after_mm, root_zone_taw_after_mm = (
                self._root_zone_water_status()
            )

            self.water_balance_trace.append(
                {
                    "day": int(self.days),
                    "year": int(self.year) if self.year is not None else -1,
                    "rainfall_mm": float(rainfall_mm),
                    "root_zone_depletion_before_mm": (
                        root_zone_depletion_before_mm
                    ),
                    "root_zone_taw_before_mm": root_zone_taw_before_mm,
                    "action_fraction": action_fraction,
                    "requested_irrigation_mm": requested_irrigation_mm,
                    "applied_irrigation_mm": float(depth),
                    "root_zone_depletion_after_mm": float(
                        root_zone_depletion_after_mm
                    ),
                    "root_zone_taw_after_mm": float(
                        root_zone_taw_after_mm
                    ),
                    "daily_stress_after": float(daily_stress),
                }
            )

            terminated = bool(
                self.model._clock_struct.model_is_finished
            )
            truncated = False

            dry_yield = 0.0
            model_irrigation: float | None = None

            if terminated:
                dry_yield, model_irrigation = self._final_stats()

            reward = self._reward(
                depth,
                terminated,
                dry_yield,
            )

            self.cumulative_reward += reward

            observation = self._validate_observation(
                self._get_obs()
            )

            info = self._info(
                final=terminated,
                dry_yield=dry_yield,
            )

            if model_irrigation is not None:
                # The action ledger remains authoritative for cap
                # enforcement. AquaCrop's value is retained for auditing.
                info["model_seasonal_irrigation_mm"] = float(
                    model_irrigation
                )

            return (
                observation,
                float(reward),
                bool(terminated),
                bool(truncated),
                info,
            )

        def get_water_balance_trace(
            self,
        ) -> list[dict[str, float | int]]:
            """Return a copy of the daily irrigation diagnostic trace."""
            return [
                row.copy()
                for row in self.water_balance_trace
            ]

        def close(self):
            self.model = None
            self.weather = None
            self.year = None
            self._has_reset = False

    return _MaizeEnv()