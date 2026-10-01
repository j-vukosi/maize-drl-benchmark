from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from maize_benchmark.config import load_benchmark_config
from maize_benchmark.envs.maize_env import make_maize_env
from maize_benchmark.envs.scenarios import load_scenario
from maize_benchmark.envs.wrappers import (
    wrap_fixed_feature_scale,
    wrap_normalized_irrigation_action,
    wrap_strict_float32_observation,
)


SplitName = Literal[
    "train",
    "val",
    "validation",
    "test",
]

_NO_DEFAULT = object()
_MISSING = object()

_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": (
        "train",
        "training",
    ),
    "val": (
        "val",
        "validation",
        "validate",
    ),
    "validation": (
        "validation",
        "val",
        "validate",
    ),
    "test": (
        "test",
        "testing",
    ),
}


def _as_mapping(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    """Convert a configuration-like object to a plain dictionary."""

    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    raise TypeError(
        f"{name} must be a mapping or configuration object, "
        f"received {type(value).__name__}."
    )


def _lookup(
    mapping: Mapping[str, Any],
    *paths: tuple[str, ...],
    default: Any = _NO_DEFAULT,
) -> Any:
    """Return the first value found among nested configuration paths.

    ``_NO_DEFAULT`` means that the caller did not provide a default and that
    a missing value should raise ``KeyError``.

    ``_MISSING`` may be passed explicitly when a caller needs to distinguish
    a missing key from a configured value of ``None``.
    """

    for path in paths:
        current: Any = mapping
        found = True

        for key in path:
            if not isinstance(current, Mapping):
                found = False
                break

            if key not in current:
                found = False
                break

            current = current[key]

        if found:
            return current

    if default is not _NO_DEFAULT:
        return default

    formatted_paths = ", ".join(
        ".".join(path)
        for path in paths
    )

    raise KeyError(
        "None of the expected configuration paths were found: "
        f"{formatted_paths}"
    )


def _resolve_option(
    *,
    scenario_config: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
    keys: Sequence[str],
    default: Any,
) -> Any:
    """Resolve one scalar option with scenario values taking precedence.

    Search order:

    1. Scenario top-level value
    2. Scenario ``environment`` section
    3. Scenario ``env`` section
    4. Benchmark top-level value
    5. Benchmark ``environment`` section
    6. Benchmark ``env`` section
    7. Supplied default
    """

    scenario_paths: list[tuple[str, ...]] = []
    benchmark_paths: list[tuple[str, ...]] = []

    for key in keys:
        scenario_paths.extend(
            (
                (key,),
                ("environment", key),
                ("env", key),
            )
        )

        benchmark_paths.extend(
            (
                (key,),
                ("protocol", key),
                ("environment", key),
                ("env", key),
            )
        )

    scenario_value = _lookup(
        scenario_config,
        *scenario_paths,
        default=_MISSING,
    )

    if scenario_value is not _MISSING:
        return scenario_value

    benchmark_value = _lookup(
        benchmark_config,
        *benchmark_paths,
        default=_MISSING,
    )

    if benchmark_value is not _MISSING:
        return benchmark_value

    return default


def _normalise_split(
    split: str,
) -> str:
    """Validate and normalize an environment split name."""

    normalised = str(split).strip().lower()

    if normalised not in _SPLIT_ALIASES:
        allowed = ", ".join(
            sorted(_SPLIT_ALIASES)
        )

        raise ValueError(
            f"Unknown environment split {split!r}. "
            f"Expected one of: {allowed}."
        )

    return normalised


def _coerce_years(
    value: Any,
    *,
    scenario: str,
    split: str,
) -> tuple[int, ...]:
    """Convert a configured year collection into a validated tuple."""

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"Years for scenario={scenario!r}, split={split!r} "
            "must be a collection, not a string."
        )

    if not isinstance(value, Sequence):
        raise TypeError(
            f"Years for scenario={scenario!r}, split={split!r} "
            f"must be a sequence, received "
            f"{type(value).__name__}."
        )

    years = tuple(
        sorted(
            {
                int(year)
                for year in value
            }
        )
    )

    if not years:
        raise ValueError(
            f"No years configured for scenario={scenario!r}, "
            f"split={split!r}."
        )

    return years


def _resolve_years(
    scenario_config: Mapping[str, Any],
    *,
    scenario: str,
    split: str,
) -> tuple[int, ...]:
    """Resolve scenario years across supported configuration layouts."""

    aliases = _SPLIT_ALIASES[split]
    paths: list[tuple[str, ...]] = []

    for alias in aliases:
        paths.extend(
            (
                ("splits", alias),
                ("years", alias),
                (alias,),
                (f"{alias}_years",),
            )
        )

    value = _lookup(
        scenario_config,
        *paths,
        default=_MISSING,
    )

    if value is _MISSING:
        expected = ", ".join(
            ".".join(path)
            for path in paths
        )

        raise KeyError(
            f"Could not resolve {split!r} years for scenario "
            f"{scenario!r}. Checked: {expected}"
        )

    return _coerce_years(
        value,
        scenario=scenario,
        split=split,
    )


def _resolve_path(
    root: Path,
    value: str | Path,
) -> Path:
    """Resolve a configured project-relative or absolute path."""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = root / path

    return path.resolve()


def _resolve_climate_file(
    root: Path,
    benchmark_config: Mapping[str, Any],
    scenario_config: Mapping[str, Any],
) -> Path:
    """Resolve the AquaCrop historical climate file."""

    configured = _lookup(
        scenario_config,
        ("climate_file",),
        ("environment", "climate_file"),
        ("weather", "climate_file"),
        default=_MISSING,
    )

    if configured is _MISSING:
        configured = _lookup(
            benchmark_config,
            ("climate_file",),
            ("environment", "climate_file"),
            ("weather", "climate_file"),
            ("paths", "climate_file"),
            default=_MISSING,
        )

    candidates: list[Path] = []

    if configured is not _MISSING and configured is not None:
        candidates.append(
            _resolve_path(
                root,
                configured,
            )
        )

    candidates.extend(
        (
            root
            / "external"
            / "aquacropgymnasium"
            / "weather_data"
            / "champion_climate.txt",
            root
            / "upstreams"
            / "aquacropgymnasium"
            / "weather_data"
            / "champion_climate.txt",
        )
    )

    for candidate in candidates:
        resolved_candidate = candidate.resolve()

        if resolved_candidate.is_file():
            return resolved_candidate

    searched = "\n".join(
        f"  - {candidate.resolve()}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not find the AquaCrop climate file. Searched:\n"
        f"{searched}"
    )


def _normalise_cap_lookup(
    value: Any,
) -> dict[int, float]:
    """Convert a water-cap configuration into year-to-cap mappings."""

    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise TypeError(
            "Water-cap lookup must be a mapping, "
            f"received {type(value).__name__}."
        )

    payload: Mapping[Any, Any] = value

    for key in (
        "cap_lookup",
        "water_caps",
        "caps",
        "year_caps",
        "irrigation_caps",
    ):
        nested = payload.get(key)

        if isinstance(nested, Mapping):
            payload = nested
            break

    result: dict[int, float] = {}

    for year, cap in payload.items():
        try:
            year_int = int(year)
            cap_float = float(cap)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid water-cap entry: "
                f"year={year!r}, cap={cap!r}."
            ) from exc

        if not np.isfinite(cap_float):
            raise ValueError(
                f"Water cap for year {year_int} must be finite, "
                f"received {cap_float}."
            )

        if cap_float < 0:
            raise ValueError(
                f"Water cap for year {year_int} cannot be negative."
            )

        result[year_int] = cap_float

    return result


def _load_cap_file(
    path: Path,
) -> dict[int, float]:
    """Load a per-year irrigation-cap JSON document."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Configured water-cap file does not exist: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    return _normalise_cap_lookup(
        payload
    )


def _resolve_cap_lookup(
    root: Path,
    scenario: str,
    scenario_config: Mapping[str, Any],
) -> dict[int, float]:
    """Resolve per-year irrigation caps for limited-water experiments."""

    inline = _lookup(
        scenario_config,
        ("cap_lookup",),
        ("water_caps",),
        ("year_caps",),
        ("irrigation", "cap_lookup"),
        ("irrigation", "water_caps"),
        default=_MISSING,
    )

    if inline is not _MISSING:
        return _normalise_cap_lookup(
            inline
        )

    configured_file = _lookup(
        scenario_config,
        ("water_caps_file",),
        ("cap_file",),
        ("irrigation", "water_caps_file"),
        ("irrigation", "cap_file"),
        default=_MISSING,
    )

    if (
        configured_file is not _MISSING
        and configured_file is not None
    ):
        return _load_cap_file(
            _resolve_path(
                root,
                configured_file,
            )
        )

    if scenario == "limited_water":
        default_candidates = (
            root
            / "scenarios"
            / "limited_water"
            / "water_caps.json",
            root
            / "configs"
            / "scenarios"
            / "limited_water"
            / "water_caps.json",
            root
            / "configs"
            / "scenarios"
            / "water_caps.json",
        )

        for candidate in default_candidates:
            resolved_candidate = candidate.resolve()

            if resolved_candidate.is_file():
                return _load_cap_file(
                    resolved_candidate
                )

        searched = "\n".join(
            f"  - {candidate.resolve()}"
            for candidate in default_candidates
        )

        raise FileNotFoundError(
            "The limited_water scenario requires a water-cap "
            "mapping or JSON file. Searched:\n"
            f"{searched}"
        )

    return {}


def _resolve_mapping_option(
    *,
    scenario_config: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
    paths: Sequence[tuple[str, ...]],
    allowed_keys: frozenset[str],
    option_name: str,
) -> dict[str, float]:
    """Merge validated numeric environment parameters.

    Scenario values override benchmark defaults. Metadata fields such as
    units, descriptions, source labels and version names are not forwarded
    to the environment constructor.
    """

    benchmark_value = _lookup(
        benchmark_config,
        *paths,
        default={},
    )

    scenario_value = _lookup(
        scenario_config,
        *paths,
        default={},
    )

    benchmark_mapping = (
        {}
        if benchmark_value is None
        else _as_mapping(
            benchmark_value,
            name=f"Benchmark {option_name}",
        )
    )

    scenario_mapping = (
        {}
        if scenario_value is None
        else _as_mapping(
            scenario_value,
            name=f"Scenario {option_name}",
        )
    )

    merged = {
        **benchmark_mapping,
        **scenario_mapping,
    }

    resolved: dict[str, float] = {}

    for key in allowed_keys:
        if key not in merged:
            continue

        value = merged[key]

        if isinstance(value, bool):
            raise TypeError(
                f"{option_name}.{key} must be numeric, "
                f"not boolean: {value!r}."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{option_name}.{key} must be numeric, "
                f"received {value!r}."
            ) from exc

        if not np.isfinite(numeric_value):
            raise ValueError(
                f"{option_name}.{key} must be finite, "
                f"received {numeric_value}."
            )

        resolved[key] = numeric_value

    return resolved


def _validate_final_environment(
    env: Any,
) -> None:
    """Validate the spaces exposed to every benchmark algorithm."""

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "Gymnasium is required to validate the benchmark "
            "environment."
        ) from exc

    observation_space = getattr(
        env,
        "observation_space",
        None,
    )

    action_space = getattr(
        env,
        "action_space",
        None,
    )

    if not isinstance(
        observation_space,
        gym.spaces.Box,
    ):
        raise TypeError(
            "Final observation space must be "
            "gymnasium.spaces.Box, received "
            f"{type(observation_space).__name__}."
        )

    if not isinstance(
        action_space,
        gym.spaces.Box,
    ):
        raise TypeError(
            "Final action space must be gymnasium.spaces.Box, "
            f"received {type(action_space).__name__}."
        )

    expected_dtype = np.dtype(
        np.float32
    )

    if np.dtype(
        observation_space.dtype
    ) != expected_dtype:
        raise TypeError(
            "Final observation space must use float32, "
            f"received {observation_space.dtype}."
        )

    if np.dtype(
        observation_space.low.dtype
    ) != expected_dtype:
        raise TypeError(
            "Observation-space low bounds must use float32, "
            f"received {observation_space.low.dtype}."
        )

    if np.dtype(
        observation_space.high.dtype
    ) != expected_dtype:
        raise TypeError(
            "Observation-space high bounds must use float32, "
            f"received {observation_space.high.dtype}."
        )

    if (
        observation_space.low.shape
        != observation_space.shape
    ):
        raise ValueError(
            "Observation-space lower-bound shape does not match "
            f"the declared shape: low={observation_space.low.shape}, "
            f"space={observation_space.shape}."
        )

    if (
        observation_space.high.shape
        != observation_space.shape
    ):
        raise ValueError(
            "Observation-space upper-bound shape does not match "
            f"the declared shape: high={observation_space.high.shape}, "
            f"space={observation_space.shape}."
        )

    if np.dtype(
        action_space.dtype
    ) != expected_dtype:
        raise TypeError(
            "Final action space must use float32, "
            f"received {action_space.dtype}."
        )

    if np.dtype(
        action_space.low.dtype
    ) != expected_dtype:
        raise TypeError(
            "Action-space low bounds must use float32, "
            f"received {action_space.low.dtype}."
        )

    if np.dtype(
        action_space.high.dtype
    ) != expected_dtype:
        raise TypeError(
            "Action-space high bounds must use float32, "
            f"received {action_space.high.dtype}."
        )

    if not np.all(
        np.isfinite(
            action_space.low
        )
    ):
        raise ValueError(
            "Action-space lower bounds contain non-finite values."
        )

    if not np.all(
        np.isfinite(
            action_space.high
        )
    ):
        raise ValueError(
            "Action-space upper bounds contain non-finite values."
        )


def build_env(
    project_root: str | Path,
    scenario: str,
    *,
    split: SplitName,
    seed: int,
    fixed_year: int | None = None,
):
    """Build the shared AquaCrop maize environment.

    Wrapper order is identical for PPO, CrossQ, DSAC-T and DreamerV3:

    1. Native irrigation depth is exposed through a shared ``[-1, 1]``
       continuous action space.
    2. The fixed 26-feature observation transformation is applied.
    3. A strict float32 observation contract is applied last.
    """

    root = Path(
        project_root
    ).resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Project root does not exist: {root}"
        )

    scenario_name = (
        str(scenario)
        .strip()
        .lower()
    )

    split_name = _normalise_split(
        split
    )

    benchmark_config = _as_mapping(
        load_benchmark_config(root),
        name="Benchmark configuration",
    )

    scenario_config = _as_mapping(
        load_scenario(
            root,
            scenario_name,
        ),
        name=(
            "Scenario configuration "
            f"{scenario_name!r}"
        ),
    )

    climate_file = _resolve_climate_file(
        root,
        benchmark_config,
        scenario_config,
    )

    years = _resolve_years(
        scenario_config,
        scenario=scenario_name,
        split=split_name,
    )

    if fixed_year is not None:
        fixed_year = int(
            fixed_year
        )

        if fixed_year not in years:
            raise ValueError(
                f"Fixed year {fixed_year} is not part of "
                f"scenario={scenario_name!r}, "
                f"split={split_name!r}. "
                f"Available years: {years}"
            )

    cap_lookup = _resolve_cap_lookup(
        root,
        scenario_name,
        scenario_config,
    )

    crop_name = str(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=(
                "crop_name",
                "crop",
            ),
            default="Maize",
        )
    )

    soil_name = str(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=(
                "soil_name",
                "soil",
            ),
            default="SandyLoam",
        )
    )

    planting_date = str(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=(
                "planting_date",
            ),
            default="05/01",
        )
    )

    season_end = str(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=(
                "season_end",
                "harvest_end",
            ),
            default="12/31",
        )
    )

    max_daily_irrigation_mm = float(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=(
                "max_daily_irrigation_mm",
                "max_irrigation_mm",
            ),
            default=25.0,
        )
    )

    if not np.isfinite(
        max_daily_irrigation_mm
    ):
        raise ValueError(
            "max_daily_irrigation_mm must be finite."
        )

    if max_daily_irrigation_mm <= 0:
        raise ValueError(
            "max_daily_irrigation_mm must be greater than zero."
        )

    min_irrigation_event_mm = float(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=("min_irrigation_event_mm",),
            default=0.0,
        )
    )

    configured_cap = _resolve_option(
        scenario_config=scenario_config,
        benchmark_config=benchmark_config,
        keys=(
            "irrigation_cap_mm",
            "seasonal_irrigation_cap_mm",
        ),
        default=None,
    )

    irrigation_cap_mm = (
        None
        if configured_cap is None
        else float(configured_cap)
    )

    if irrigation_cap_mm is not None:
        if not np.isfinite(
            irrigation_cap_mm
        ):
            raise ValueError(
                "irrigation_cap_mm must be finite."
            )

        if irrigation_cap_mm < 0:
            raise ValueError(
                "irrigation_cap_mm cannot be negative."
            )

    action_mode = str(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=(
                "action_mode",
            ),
            default="continuous_depth",
        )
    )

    reward_version = str(
        _resolve_option(
            scenario_config=scenario_config,
            benchmark_config=benchmark_config,
            keys=(
                "reward_version",
            ),
            default="balanced_v2",
        )
    )

    reward_config = _resolve_mapping_option(
        scenario_config=scenario_config,
        benchmark_config=benchmark_config,
        paths=(
            ("reward_config",),
            ("reward",),
            (
                "environment",
                "reward_config",
            ),
        ),
        allowed_keys=frozenset(
            {
                "yield_power",
                "yield_scale",
                "water_penalty_per_mm",
                "stress_penalty_per_day",
            }
        ),
        option_name="reward_config",
    )

    profit_config = _resolve_mapping_option(
        scenario_config=scenario_config,
        benchmark_config=benchmark_config,
        paths=(
            ("profit_config",),
            ("profit",),
            ("economics",),
            (
                "environment",
                "profit_config",
            ),
        ),
        allowed_keys=frozenset(
            {
                "maize_price_per_tonne",
                "irrigation_cost_per_mm_ha",
                "fixed_cost_per_ha",
            }
        ),
        option_name="profit_config",
    )

    # The upstream base environment may construct an intermediate Box with
    # float64 bound arrays while declaring float32. These two construction
    # notices are limited to that inner environment. The final wrapped spaces
    # are rebuilt and validated below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"WARN: Box low's precision "
                r"lowered by casting to float32.*"
            ),
            category=UserWarning,
        )

        warnings.filterwarnings(
            "ignore",
            message=(
                r"WARN: Box high's precision "
                r"lowered by casting to float32.*"
            ),
            category=UserWarning,
        )

        env = make_maize_env(
            climate_file=climate_file,
            years=years,
            seed=int(seed),
            crop_name=crop_name,
            soil_name=soil_name,
            planting_date=planting_date,
            season_end=season_end,
            max_daily_irrigation_mm=max_daily_irrigation_mm,
            min_irrigation_event_mm=min_irrigation_event_mm,
            irrigation_cap_mm=irrigation_cap_mm,
            cap_lookup=cap_lookup,
            action_mode=action_mode,
            reward_version=reward_version,
            reward_config=reward_config,
            profit_config=profit_config,
            fixed_year=fixed_year,
        )

    try:
        env = wrap_normalized_irrigation_action(
            env
        )

        env = wrap_fixed_feature_scale(
            env
        )

        # This must remain the outermost observation wrapper.
        env = wrap_strict_float32_observation(
            env
        )

        _validate_final_environment(
            env
        )

    except Exception:
        env.close()
        raise

    return env