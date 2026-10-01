from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maize_benchmark.config import load_benchmark_config, load_yaml
from maize_benchmark.envs.factory import build_env
from maize_benchmark.evaluation import evaluate_policy, save_evaluation
from maize_benchmark.logging_utils import ResourceTracker, ensure_dir, system_manifest, write_json
from maize_benchmark.seeding import seed_everything


class EpisodeCSVCallback:
    """Factory for a Stable-Baselines3 callback without importing SB3 at module import."""

    @staticmethod
    def make(path: Path, resource_tracker: ResourceTracker):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.rows: list[dict[str, Any]] = []

            def _on_step(self) -> bool:
                resource_tracker.sample()
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for info, done in zip(infos, dones):
                    if done and info.get("final"):
                        self.rows.append(
                            {
                                "timesteps": int(self.num_timesteps),
                                "episode_reward": info.get("episode_reward"),
                                "dry_yield_tonne_per_ha": info.get(
                                    "dry_yield_tonne_per_ha"
                                ),
                                "total_irrigation_mm": info.get("total_irrigation_mm"),
                                "profit": info.get("profit"),
                                "water_productivity_kg_per_ha_per_mm": info.get(
                                    "water_productivity_kg_per_ha_per_mm"
                                ),
                                "stress_exposure": info.get("stress_exposure"),
                                "stress_days": info.get("stress_days"),
                                "max_daily_stress": info.get("max_daily_stress"),
                                "year": info.get("year"),
                            }
                        )
                return True

            def _on_training_end(self) -> None:
                pd.DataFrame(self.rows).to_csv(path, index=False)

        return _Callback()


def _activation(name: str):
    import torch.nn as nn

    lookup = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
    try:
        return lookup[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name}") from exc


def train_sb3(
    *,
    project_root: str | Path,
    algorithm: str,
    scenario: str,
    seed: int,
    total_timesteps: int,
) -> Path:
    root = Path(project_root)
    seed_everything(seed)
    benchmark = load_benchmark_config(root)
    cfg = load_yaml(root / "configs" / "algorithms" / f"{algorithm}.yaml")
    run_dir = ensure_dir(
        root / benchmark["output_root"] / "runs" / algorithm / scenario / f"seed_{seed}"
    )
    write_json(run_dir / "system_manifest.json", system_manifest())
    write_json(run_dir / "resolved_config.json", {"benchmark": benchmark, "algorithm": cfg})

    env = build_env(root, scenario, split="train", seed=seed)
    monitor_path = run_dir / "monitor"
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(env, filename=str(monitor_path))

    if algorithm == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            policy_kwargs={
                "net_arch": dict(pi=cfg["policy_hidden"], vf=cfg["policy_hidden"]),
                "activation_fn": _activation(cfg["activation"]),
            },
            seed=seed,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=benchmark.get("compute", {}).get("device", "auto"),
        )
    elif algorithm == "crossq":
        from sb3_contrib import CrossQ

        model = CrossQ(
            "MlpPolicy",
            env,
            learning_rate=cfg["learning_rate"],
            buffer_size=cfg["buffer_size"],
            learning_starts=cfg["learning_starts"],
            batch_size=cfg["batch_size"],
            gamma=cfg["gamma"],
            train_freq=cfg["train_freq"],
            gradient_steps=cfg["gradient_steps"],
            ent_coef=cfg["ent_coef"],
            policy_delay=cfg.get("policy_delay", 3),
            policy_kwargs={
                "net_arch": dict(pi=cfg["actor_hidden"], qf=cfg["critic_hidden"]),
                "batch_norm": cfg["batch_norm"],
            },
            seed=seed,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=benchmark.get("compute", {}).get("device", "auto"),
        )
    else:
        raise ValueError("train_sb3 supports only ppo and crossq")

    with ResourceTracker() as tracker:
        callback = EpisodeCSVCallback.make(run_dir / "training_episodes.csv", tracker)
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    actual_steps = int(model.num_timesteps)
    write_json(
        run_dir / "interaction_budget.json",
        {
            "target_environment_steps": int(total_timesteps),
            "actual_environment_steps": actual_steps,
            "overshoot_steps": actual_steps - int(total_timesteps),
        },
    )
    model.save(run_dir / "model")
    write_json(run_dir / "compute.json", tracker.as_dict())
    env.close()
    return run_dir


def evaluate_sb3(
    *,
    project_root: str | Path,
    algorithm: str,
    scenario: str,
    seed: int,
    episodes_per_year: int,
    split: str = "test",
) -> Path:
    root = Path(project_root)
    benchmark = load_benchmark_config(root)
    run_dir = root / benchmark["output_root"] / "runs" / algorithm / scenario / f"seed_{seed}"
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(run_dir / "model")
    elif algorithm == "crossq":
        from sb3_contrib import CrossQ

        model = CrossQ.load(run_dir / "model")
    else:
        raise ValueError("evaluate_sb3 supports only ppo and crossq")

    def policy(obs: np.ndarray, deterministic: bool) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    frame = evaluate_policy(
        project_root=root,
        algorithm=algorithm,
        scenario=scenario,
        seed=seed,
        policy=policy,
        episodes_per_year=episodes_per_year,
        deterministic=True,
        split=split,
    )
    target = run_dir / ("evaluation.csv" if split == "test" else "validation_evaluation.csv")
    save_evaluation(frame, target)
    return target
