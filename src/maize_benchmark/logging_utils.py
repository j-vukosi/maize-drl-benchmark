from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def package_freeze() -> list[str]:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
        )
        return sorted(line.strip() for line in output.splitlines() if line.strip())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return [f"pip-freeze-failed: {exc}"]


def system_manifest() -> dict[str, Any]:
    process = psutil.Process()
    manifest: dict[str, Any] = {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "process_rss_bytes": process.memory_info().rss,
        "packages": package_freeze(),
    }
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["cuda_version"] = torch.version.cuda
        manifest["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        manifest["torch"] = None
    try:
        import tensorflow as tf

        manifest["tensorflow"] = tf.__version__
        manifest["tensorflow_gpus"] = [
            device.name for device in tf.config.list_physical_devices("GPU")
        ]
    except ImportError:
        manifest["tensorflow"] = None
    return manifest


class ResourceTracker:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.started = 0.0
        self.peak_rss = 0
        self.peak_process_tree_rss = 0

    def __enter__(self) -> "ResourceTracker":
        self.started = time.perf_counter()
        self.sample()
        return self

    def sample(self) -> None:
        try:
            main_rss = self.process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            main_rss = 0
        tree_rss = main_rss
        try:
            children = self.process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
        for child in children:
            try:
                tree_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_rss = max(self.peak_rss, main_rss)
        self.peak_process_tree_rss = max(self.peak_process_tree_rss, tree_rss)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.sample()
        self.elapsed_seconds = time.perf_counter() - self.started

    def as_dict(self) -> dict[str, float | int]:
        return {
            "wall_clock_seconds": float(getattr(self, "elapsed_seconds", 0.0)),
            "peak_rss_bytes": int(self.peak_rss),
            "peak_process_tree_rss_bytes": int(self.peak_process_tree_rss),
        }
