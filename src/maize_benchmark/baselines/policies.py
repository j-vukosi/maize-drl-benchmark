"""Legacy *gated heuristic* policies retained only for backwards compatibility.

The benchmark's conventional reference evaluation now uses the AquaCrop
native irrigation methods via baselines.runner rather than these wrappers.
These wrappers pass through the learned-agent 40% depletion gate and must NOT
be described as the Alkaff et al. (2025) conventional baselines.
"""


import numpy as np


class RainfedPolicy:
    def reset_episode(self) -> None:
        pass

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.array([-1.0], dtype=np.float32)


class SoilMoistureThresholdPolicy:
    def __init__(self, threshold: float = 0.45):
        self.threshold = float(threshold)

    def reset_episode(self) -> None:
        pass

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # Features 3 and 4 are depletion and TAW. Their fixed scale is identical,
        # so the ratio is unchanged by the shared observation transform.
        taw = max(float(obs[4]), 1e-6)
        depletion_fraction = float(obs[3]) / taw
        normalized_action = 1.0 if depletion_fraction >= self.threshold else -1.0
        return np.array([normalized_action], dtype=np.float32)


class FixedIntervalPolicy:
    def __init__(self, interval_days: int = 7):
        self.interval_days = int(interval_days)
        self.day = 0

    def reset_episode(self) -> None:
        self.day = 0

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.day += 1
        normalized_action = 1.0 if self.day % self.interval_days == 0 else -1.0
        return np.array([normalized_action], dtype=np.float32)


def make_baseline(name: str):
    if name == "rainfed":
        return RainfedPolicy()
    if name == "soil_moisture_threshold":
        return SoilMoistureThresholdPolicy()
    if name == "fixed_interval":
        return FixedIntervalPolicy()
    raise ValueError(f"Unknown baseline: {name}")
