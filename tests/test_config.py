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


def test_revised_500k_protocol_has_exact_algorithm_sampling_units():
    from maize_benchmark.config import load_yaml

    project_root = Path(__file__).resolve().parents[1]
    benchmark = load_benchmark_config(project_root)
    ppo = load_yaml(project_root / "configs" / "algorithms" / "ppo.yaml")
    dreamer = load_yaml(project_root / "configs" / "algorithms" / "dreamerv3.yaml")

    target = int(benchmark["protocol"]["total_timesteps"])
    assert target == 500_000
    assert benchmark["output_root"] == "outputs_500000_resource_audited_v3"

    assert target % int(ppo["n_steps"]) == 0
    assert int(ppo["n_steps"]) % int(ppo["batch_size"]) == 0
    assert ppo["learning_rate"] == 0.000634
    assert ppo["n_epochs"] == 23
    assert ppo["gamma"] == 0.98
    assert ppo["clip_range"] == 0.22
    assert ppo["policy_hidden"] == [64, 64]
    assert ppo["activation"] == "relu"
    assert int(dreamer["min_sample_timesteps_per_iteration"]) % int(
        dreamer["rollout_fragment_length"]
    ) == 0
    assert target % int(dreamer["min_sample_timesteps_per_iteration"]) == 0


def test_final_protocol_uses_requested_three_seeds():
    project_root = Path(__file__).resolve().parents[1]
    benchmark = load_benchmark_config(project_root)
    assert benchmark["protocol"]["seeds"] == [1, 2, 3]
