from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maize_benchmark.config import load_yaml


@dataclass(frozen=True)
class Scenario:
    name: str
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    irrigation_cap_mm: float | None
    cap_lookup: dict[int, float]
    raw: dict[str, Any]

    def cap_for_year(self, year: int) -> float | None:
        if year in self.cap_lookup:
            return float(self.cap_lookup[year])
        return self.irrigation_cap_mm


def _load_years(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing scenario split file: {path}")
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    years = tuple(int(value) for value in values)
    if not years:
        raise ValueError(f"Scenario split file is empty: {path}")
    if len(set(years)) != len(years):
        raise ValueError(f"Scenario split contains duplicate years: {path}")
    return years


def _load_scenario_manifest(scenario_dir: Path, name: str) -> dict[str, Any]:
    """Read the checked-in manifest, with support for legacy YAML layouts."""

    manifest_path = scenario_dir / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise TypeError(f"Scenario manifest must contain an object: {manifest_path}")
        manifest_name = str(payload.get("name", name)).strip().lower()
        if manifest_name != name:
            raise ValueError(
                f"Scenario manifest name mismatch: requested {name!r}, "
                f"found {manifest_name!r} in {manifest_path}."
            )
        return payload

    legacy_path = scenario_dir.parent.parent / "configs" / "scenarios" / f"{name}.yaml"
    if legacy_path.is_file():
        payload = load_yaml(legacy_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Scenario configuration must contain a mapping: {legacy_path}")
        return payload

    raise FileNotFoundError(
        f"No scenario manifest or legacy scenario YAML found for {name!r}. "
        f"Expected {manifest_path} or {legacy_path}."
    )


def load_scenario(project_root: str | Path, name: str) -> Scenario:
    root = Path(project_root).resolve()
    scenario_name = str(name).strip().lower()
    if not scenario_name or any(part in scenario_name for part in ("/", "\\", "..")):
        raise ValueError(f"Invalid scenario name: {name!r}")

    scenario_dir = root / "scenarios" / scenario_name
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario directory does not exist: {scenario_dir}")

    raw = _load_scenario_manifest(scenario_dir, scenario_name)

    # The year split JSON files are the actual frozen split artefacts. Verify
    # that any copy in the manifest agrees with them rather than silently
    # selecting a different year partition.
    split_years: dict[str, tuple[int, ...]] = {}
    for split in ("train", "validation", "test"):
        year_file = scenario_dir / f"{split}_years.json"
        if scenario_name in {"baseline_full", "baseline_full_limited_water"} and split == "validation":
            if not year_file.is_file():
                raise FileNotFoundError(f"Missing scenario split file: {year_file}")
            if __import__("json").loads(year_file.read_text(encoding="utf-8-sig")) != []:
                raise ValueError("validation years must be empty")
            split_years[split] = ()
        else:
            split_years[split] = _load_years(year_file)
        manifest_values = raw.get(f"{split}_years")
        if manifest_values is not None:
            manifest_years = tuple(int(value) for value in manifest_values)
            if set(manifest_years) != set(split_years[split]):
                raise ValueError(
                    f"{scenario_name} {split} years differ between manifest.json "
                    f"and {split}_years.json."
                )

    all_years = [set(values) for values in split_years.values()]
    if all_years[0] & all_years[1] or all_years[0] & all_years[2] or all_years[1] & all_years[2]:
        raise ValueError(f"Scenario {scenario_name!r} has overlapping year splits.")

    cap_lookup: dict[int, float] = {}
    cap_path = scenario_dir / "water_caps.json"
    if cap_path.is_file():
        cap_payload = json.loads(cap_path.read_text(encoding="utf-8-sig"))
        if not isinstance(cap_payload, dict):
            raise TypeError(f"Water-cap file must contain an object: {cap_path}")
        cap_lookup = {int(year): float(cap) for year, cap in cap_payload.items()}
        for year, cap in cap_lookup.items():
            if not math.isfinite(cap) or cap < 0:
                raise ValueError(
                    f"Water cap for year {year} must be finite and non-negative."
                )
        # Factory reads this inline mapping. This avoids requiring a second,
        # non-existent configs/scenarios YAML tree.
        raw["cap_lookup"] = cap_lookup

    water_cfg = raw.get("water", {})
    if not isinstance(water_cfg, dict):
        water_cfg = {}
    cap = water_cfg.get("irrigation_cap_mm", raw.get("irrigation_cap_mm"))

    if scenario_name == "limited_water" and not cap_lookup and cap is None:
        raise FileNotFoundError(
            "The limited_water scenario requires the checked-in "
            f"per-year SMT-80-derived allocation file: {cap_path}"
        )

    if scenario_name == "limited_water":
        required_cap_years = set().union(*all_years)
        missing_cap_years = sorted(required_cap_years - set(cap_lookup))
        if missing_cap_years:
            raise ValueError(
                "The limited_water cap table is missing scenario years: "
                f"{missing_cap_years}"
            )

    # A legacy Windows-style source path appears in the manifests. Prefer the
    # portable checked-in climate path, but keep manifest metadata intact.
    climate_file = raw.get("climate_file")
    if isinstance(climate_file, str):
        raw["climate_file"] = climate_file.replace("\\", "/")

    return Scenario(
        name=scenario_name,
        train_years=split_years["train"],
        validation_years=split_years["validation"],
        test_years=split_years["test"],
        irrigation_cap_mm=None if cap is None else float(cap),
        cap_lookup=cap_lookup,
        raw=raw,
    )
