from types import SimpleNamespace

import numpy as np
import torch

from maize_benchmark.algorithms.dsac_t import DSACTAgent
from maize_benchmark.config import load_yaml


def test_dsac_t_update_is_finite():
    cfg = load_yaml("configs/algorithms/dsac_t.yaml")
    action_space = SimpleNamespace(
        shape=(1,),
        low=np.array([0.0], dtype=np.float32),
        high=np.array([25.0], dtype=np.float32),
    )
    agent = DSACTAgent(26, action_space, cfg, torch.device("cpu"))
    batch_size = 32
    batch = {
        "obs": torch.randn(batch_size, 26),
        "act": torch.rand(batch_size, 1) * 25.0,
        "rew": torch.randn(batch_size, 1),
        "obs2": torch.randn(batch_size, 26),
        "done": torch.zeros(batch_size, 1),
    }
    metrics = agent.update(batch)
    for value in metrics.__dict__.values():
        assert np.isfinite(value)
    action = agent.act(np.zeros(26, dtype=np.float32), deterministic=True)
    assert action.shape == (1,)
    assert 0.0 <= action[0] <= 25.0
