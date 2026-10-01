from pathlib import Path

from maize_benchmark.config import deep_merge, load_benchmark_config


def test_deep_merge_preserves_nested_defaults():
    result = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert result == {"a": {"b": 3, "c": 2}}


def test_output_root_override_is_storage_only(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "benchmark.yaml").write_text(
        "output_root: outputs\nprotocol:\n  total_timesteps: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAIZE_BENCHMARK_OUTPUT_ROOT", "outputs_smoke")
    loaded = load_benchmark_config(tmp_path)
    assert loaded["output_root"] == "outputs_smoke"
    assert loaded["protocol"]["total_timesteps"] == 10
