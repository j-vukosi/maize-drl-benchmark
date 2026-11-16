#!/usr/bin/env python3
"""Regenerate benchmark tables and figures from saved evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from maize_benchmark.config import resolve_project_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    project_root = resolve_project_root(args.project_root)

    from maize_benchmark.reporting.report import generate_report

    for name, path in generate_report(project_root).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
