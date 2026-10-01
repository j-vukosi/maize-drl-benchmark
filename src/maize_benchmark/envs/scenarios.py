from __future__ import annotations

import json
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
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/prepare_scenarios.py")
    values = json.loads(path.read_text(encoding="utf-8"))
    return tuple(int(value) for value in values)


def load_scenario(project_root: str | Path, name: str) -> Scenario:
    root = Path(project_root)
    cfg = load_yaml(root / "configs" / "scenarios" / f"{name}.yaml")
    scenario_dir = root / "scenarios" / name
    train_years = _load_years(scenario_dir / "train_years.json")
    validation_years = _load_years(scenario_dir / "validation_years.json")
    test_years = _load_years(scenario_dir / "test_years.json")
    water_cfg = cfg.get("water", {})
    cap = water_cfg.get("irrigation_cap_mm")
    cap_lookup: dict[int, float] = {}
    cap_path = water_cfg.get("cap_lookup_file")
    if cap_path:
        resolved = root / cap_path
        if resolved.exists():
            raw_caps = json.loads(resolved.read_text(encoding="utf-8"))
            cap_lookup = {int(k): float(v) for k, v in raw_caps.items()}
    if name == "limited_water" and not cap_lookup and cap is None:
        raise FileNotFoundError(
            "The limited_water scenario needs calibrated caps. Run: "
            "python scripts/calibrate_water_caps.py"
        )
    return Scenario(
        name=name,
        train_years=train_years,
        validation_years=validation_years,
        test_years=test_years,
        irrigation_cap_mm=None if cap is None else float(cap),
        cap_lookup=cap_lookup,
        raw=cfg,
    )
