from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_COUNT = 26


def require_gymnasium():
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "Install Gymnasium with: pip install -e ."
        ) from exc

    return gym


def _reshape_action(
    action: Any,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Convert an action to float32 while enforcing the expected size."""

    array = np.asarray(action, dtype=np.float32)

    expected_size = int(np.prod(target_shape))
    actual_size = int(array.size)

    if actual_size != expected_size:
        raise ValueError(
            f"Action contains {actual_size} values, "
            f"but {expected_size} were expected for shape "
            f"{target_shape}."
        )

    return np.ascontiguousarray(
        array.reshape(target_shape),
        dtype=np.float32,
    )


def normalized_to_native_action(
    action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map a shared [-1, 1] action to the native environment range."""

    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)

    if low_array.shape != high_array.shape:
        raise ValueError(
            "Native action bounds have different shapes: "
            f"low={low_array.shape}, high={high_array.shape}."
        )

    normalized = _reshape_action(
        action,
        low_array.shape,
    )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            f"Normalized action contains non-finite values: {normalized}"
        )

    normalized = np.clip(
        normalized,
        -1.0,
        1.0,
    )

    native = (
        low_array
        + 0.5
        * (normalized + 1.0)
        * (high_array - low_array)
    )

    return np.ascontiguousarray(
        native,
        dtype=np.float32,
    )


def native_to_normalized_action(
    action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Invert normalized_to_native_action with safe clipping."""

    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)

    if low_array.shape != high_array.shape:
        raise ValueError(
            "Native action bounds have different shapes: "
            f"low={low_array.shape}, high={high_array.shape}."
        )

    native = _reshape_action(
        action,
        low_array.shape,
    )

    if not np.all(np.isfinite(native)):
        raise ValueError(
            f"Native action contains non-finite values: {native}"
        )

    span = np.maximum(
        high_array - low_array,
        np.finfo(np.float32).eps,
    )

    normalized = (
        2.0 * (native - low_array) / span - 1.0
    )

    return np.ascontiguousarray(
        np.clip(normalized, -1.0, 1.0),
        dtype=np.float32,
    )


def wrap_normalized_irrigation_action(env):
    """Expose a shared [-1, 1] action interface.

    RLlib DreamerV3's continuous actor emits tanh-bounded actions around
    [-1, 1]. Giving every learned policy this same normalized interface
    avoids an algorithm-specific transform. The wrapped AquaCrop
    environment still receives irrigation depth in millimetres.
    """

    gym = require_gymnasium()

    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(
            "Normalized irrigation wrapper requires a Box action space."
        )

    native_low = np.asarray(
        env.action_space.low,
        dtype=np.float32,
    ).copy()

    native_high = np.asarray(
        env.action_space.high,
        dtype=np.float32,
    ).copy()

    if native_low.shape != native_high.shape:
        raise ValueError(
            "Environment action bounds have inconsistent shapes."
        )

    class _Wrapper(gym.ActionWrapper):
        def __init__(self, inner):
            super().__init__(inner)

            self.native_action_low = native_low.copy()
            self.native_action_high = native_high.copy()

            normalized_low = np.full(
                inner.action_space.shape,
                -1.0,
                dtype=np.float32,
            )
            normalized_high = np.full(
                inner.action_space.shape,
                1.0,
                dtype=np.float32,
            )

            self.action_space = gym.spaces.Box(
                low=normalized_low,
                high=normalized_high,
                dtype=np.float32,
            )

        def action(self, action):
            return normalized_to_native_action(
                action,
                self.native_action_low,
                self.native_action_high,
            )

        def reverse_action(self, action):
            return native_to_normalized_action(
                action,
                self.native_action_low,
                self.native_action_high,
            )

    return _Wrapper(env)


class FixedFeatureScale:
    """Shared, non-learning feature transform for all four agents.

    Source observation order:

    0. age
    1. canopy cover
    2. biomass
    3. depletion
    4. total available water
    5-11. seven precipitation observations
    12-18. seven minimum-temperature observations
    19-25. seven maximum-temperature observations
    """

    scale = np.array(
        [
            365.0,
            1.0,
            50.0,
            500.0,
            500.0,
        ]
        + [100.0] * 7
        + [50.0] * 7
        + [50.0] * 7,
        dtype=np.float32,
    )

    offset = np.zeros(
        FEATURE_COUNT,
        dtype=np.float32,
    )

    @classmethod
    def apply(
        cls,
        observation: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(
            observation,
            dtype=np.float32,
        )

        if array.shape[-1] != FEATURE_COUNT:
            raise ValueError(
                f"Expected {FEATURE_COUNT} observation features, "
                f"received shape {array.shape}."
            )

        if not np.all(np.isfinite(array)):
            invalid_indices = np.argwhere(
                ~np.isfinite(array)
            )

            raise ValueError(
                "Observation contains non-finite values at positions "
                f"{invalid_indices.tolist()}."
            )

        transformed = (
            array - cls.offset
        ) / cls.scale

        transformed = np.clip(
            transformed,
            -10.0,
            10.0,
        )

        return np.ascontiguousarray(
            transformed,
            dtype=np.float32,
        )


def wrap_fixed_feature_scale(env):
    """Apply the fixed 26-feature scaling transformation."""

    gym = require_gymnasium()

    expected_shape = (FEATURE_COUNT,)

    if env.observation_space.shape != expected_shape:
        raise ValueError(
            "Fixed feature scaling requires observation shape "
            f"{expected_shape}, received "
            f"{env.observation_space.shape}."
        )

    class _Wrapper(gym.ObservationWrapper):
        def __init__(self, inner):
            super().__init__(inner)

            observation_low = np.full(
                expected_shape,
                -10.0,
                dtype=np.float32,
            )
            observation_high = np.full(
                expected_shape,
                10.0,
                dtype=np.float32,
            )

            self.observation_space = gym.spaces.Box(
                low=observation_low,
                high=observation_high,
                dtype=np.float32,
            )

        def observation(self, observation):
            transformed = FixedFeatureScale.apply(
                observation
            )

            if transformed.shape != self.observation_space.shape:
                raise ValueError(
                    "Transformed observation shape mismatch: "
                    f"received {transformed.shape}, expected "
                    f"{self.observation_space.shape}."
                )

            if not self.observation_space.contains(transformed):
                raise ValueError(
                    "Transformed observation is outside the declared "
                    "fixed-scale observation space."
                )

            return transformed

    return _Wrapper(env)

def wrap_strict_float32_observation(env):
    """Enforce the final benchmark observation contract.

    All algorithms must receive contiguous float32 observations whose shape
    and values agree with the declared Gymnasium Box space.
    """

    gym = require_gymnasium()

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            "The strict float32 wrapper requires a Box observation space, "
            f"received {type(env.observation_space).__name__}."
        )

    source_low = np.asarray(
        env.observation_space.low,
        dtype=np.float32,
    )
    source_high = np.asarray(
        env.observation_space.high,
        dtype=np.float32,
    )

    if source_low.shape != source_high.shape:
        raise ValueError(
            "Observation-space bounds have different shapes: "
            f"low={source_low.shape}, high={source_high.shape}."
        )

    class _StrictFloat32Observation(gym.ObservationWrapper):
        def __init__(self, inner):
            super().__init__(inner)

            self.observation_space = gym.spaces.Box(
                low=np.ascontiguousarray(
                    source_low.copy(),
                    dtype=np.float32,
                ),
                high=np.ascontiguousarray(
                    source_high.copy(),
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )

        def observation(self, observation):
            obs = np.ascontiguousarray(
                np.asarray(observation, dtype=np.float32)
            )

            if obs.shape != self.observation_space.shape:
                raise ValueError(
                    "Observation shape mismatch: "
                    f"received {obs.shape}, expected "
                    f"{self.observation_space.shape}."
                )

            if not np.all(np.isfinite(obs)):
                invalid = np.argwhere(~np.isfinite(obs))

                raise ValueError(
                    "Observation contains non-finite values at positions "
                    f"{invalid.tolist()}."
                )

            if not self.observation_space.contains(obs):
                below = np.argwhere(
                    obs < self.observation_space.low
                )
                above = np.argwhere(
                    obs > self.observation_space.high
                )

                raise ValueError(
                    "Observation is outside the declared observation space.\n"
                    f"Below-bound positions: {below.tolist()}\n"
                    f"Above-bound positions: {above.tolist()}\n"
                    f"Observation: {obs}"
                )

            return obs

    return _StrictFloat32Observation(env)