from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when benchmark configuration is inconsistent."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping in {path}")
    return data



def load_benchmark_config(project_root: str | Path) -> dict[str, Any]:
    """Load benchmark.yaml and apply the optional run-output override.

    ``MAIZE_BENCHMARK_OUTPUT_ROOT`` is used by the Windows orchestration script
    to isolate smoke-test artifacts from honours-run artifacts. The scientific
    protocol remains frozen in YAML; only the storage location changes.
    """

    root = Path(project_root)
    data = load_yaml(root / "configs" / "benchmark.yaml")
    override = os.environ.get("MAIZE_BENCHMARK_OUTPUT_ROOT", "").strip()
    if override:
        data = dict(data)
        data["output_root"] = override
    return data

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class Paths:
    root: Path
    config_dir: Path
    scenario_dir: Path
    output_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "Paths":
        root_path = Path(root).resolve()
        return cls(
            root=root_path,
            config_dir=root_path / "configs",
            scenario_dir=root_path / "scenarios",
            output_dir=root_path / "outputs",
        )


def resolve_project_root(start: str | Path | None = None) -> Path:
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "configs" / "benchmark.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not find configs/benchmark.yaml from current path")
