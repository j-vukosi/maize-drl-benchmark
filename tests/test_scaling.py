import numpy as np

from maize_benchmark.envs.wrappers import FixedFeatureScale


def test_fixed_feature_scaling_shape_and_ratio():
    obs = np.zeros(26, dtype=np.float32)
    obs[0] = 182.5
    obs[3] = 100.0
    obs[4] = 200.0
    obs[5:12] = 50.0
    scaled = FixedFeatureScale.apply(obs)
    assert scaled.shape == (26,)
    assert scaled.dtype == np.float32
    assert np.isclose(scaled[0], 0.5)
    assert np.isclose(scaled[3] / scaled[4], 0.5)
    assert np.all(np.isfinite(scaled))


def test_scaling_rejects_wrong_feature_count():
    try:
        FixedFeatureScale.apply(np.zeros(25, dtype=np.float32))
    except ValueError as exc:
        assert "26" in str(exc)
    else:
        raise AssertionError("Expected a feature-count error")


def test_normalized_action_round_trip():
    from maize_benchmark.envs.wrappers import (
        native_to_normalized_action,
        normalized_to_native_action,
    )

    low = np.array([0.0], dtype=np.float32)
    high = np.array([25.0], dtype=np.float32)
    normalized = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    mapped = np.array(
        [normalized_to_native_action([value], low, high)[0] for value in normalized]
    )
    assert np.allclose(mapped, [0.0, 12.5, 25.0])
    recovered = np.array(
        [native_to_normalized_action([value], low, high)[0] for value in mapped]
    )
    assert np.allclose(recovered, normalized)
