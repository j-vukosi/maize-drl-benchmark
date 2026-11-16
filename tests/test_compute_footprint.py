"""Regression checks for CrossQ CPU/replay budget and DreamerV3 disk usage."""
from pathlib import Path

import pytest

from maize_benchmark.algorithms.dreamer_runner import (
    _read_final_checkpoint,
    _save_rolling_checkpoint,
)
from maize_benchmark.algorithms.sb3_runner import replay_capacity_for_budget
from maize_benchmark.config import load_benchmark_config, load_yaml


class FakeDreamer:
    def __init__(self):
        self.fail = False

    def save_to_path(self, target):
        if self.fail:
            raise RuntimeError("simulated incomplete checkpoint")
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoint-content").write_text("model")
        return target


def test_crossq_replay_cap_does_not_drop_any_of_500k_steps():
    assert replay_capacity_for_budget(1_000_000, 500_000, 256) == 500_000
    assert replay_capacity_for_budget(1_000_000, 6_000, 256) == 6_000
    assert replay_capacity_for_budget(15_000, 500_000, 256) == 15_000
    with pytest.raises(ValueError):
        replay_capacity_for_budget(10, 500_000, 256)


def test_rolling_checkpoint_keeps_single_intermediate(tmp_path):
    algo = FakeDreamer()
    previous = None
    for step in (50_000, 100_000, 150_000):
        previous = _save_rolling_checkpoint(algo, tmp_path, step, previous)
    assert previous.name == "step_150000"
    assert list((tmp_path / "checkpoints").iterdir()) == [previous]
    assert (previous / "checkpoint-content").read_text() == "model"


def test_checkpoint_save_failure_keeps_recovery_snapshot(tmp_path):
    algo = FakeDreamer()
    previous = _save_rolling_checkpoint(algo, tmp_path, 50_000, None)
    algo.fail = True
    with pytest.raises(RuntimeError, match="simulated"):
        _save_rolling_checkpoint(algo, tmp_path, 100_000, previous)
    assert (previous / "checkpoint-content").exists()


def test_final_checkpoint_reads_after_moving_run_directory(tmp_path):
    import json
    path = tmp_path / "run" / "checkpoint_final"
    path.mkdir(parents=True)
    (path / "checkpoint-content").write_text("model")
    (tmp_path / "run" / "checkpoints.json").write_text(
        json.dumps([{"environment_steps": 500_000, "path": "checkpoint_final"}]),
        encoding="utf-8",
    )
    assert _read_final_checkpoint(tmp_path / "run") == str(path)


def test_scripts_preserve_500k_three_seed_contract_and_scenario_order():
    root = Path(__file__).resolve().parents[1]
    bench = load_benchmark_config(root)
    crossq = load_yaml(root / "configs" / "algorithms" / "crossq.yaml")
    dreamer = load_yaml(root / "configs" / "algorithms" / "dreamerv3.yaml")
    script = (root / "scripts" / "run_benchmark.sh").read_text()
    assert bench["protocol"]["total_timesteps"] == 500_000
    assert bench["protocol"]["seeds"] == [1, 2, 3]
    assert bench["scenarios"] == ["extreme_dry", "dry", "normal", "limited_water", "wet"]
    assert "TIMESTEPS=500000" in script
    assert "TIMESTEPS=600000" not in script
    assert script.count('"$PYTHON" scripts/generate_report.py') == 1
    assert crossq["critic_hidden"] == [1024, 1024]  # no silent change in algorithm
    assert crossq["gradient_steps"] == 1
    assert dreamer["training_ratio"] == 32
    assert dreamer["checkpoint_interval_steps"] == 50_000
    assert dreamer["checkpoint_retention"] == "latest_intermediate_only"
