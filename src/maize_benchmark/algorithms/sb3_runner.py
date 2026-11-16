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
    def make(path: Path, resource_tracker: ResourceTracker, *, sample_every_steps: int = 250):
        from stable_baselines3.common.callbacks import BaseCallback

        if sample_every_steps < 1:
            raise ValueError("sample_every_steps must be positive")

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.rows: list[dict[str, Any]] = []

            def _on_step(self) -> bool:
                # ResourceTracker walks the process tree; do not poll it on
                # every environment interaction (500,000 times per run).
                if self.num_timesteps % sample_every_steps == 0:
                    resource_tracker.sample()
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for info, done in zip(infos, dones):
                    if done and info.get("final"):
                        resource_tracker.sample()
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


def replay_capacity_for_budget(configured: int, total_timesteps: int, batch_size: int) -> int:
    """Retain every observed transition without allocating unreachable slots."""
    if min(configured, total_timesteps, batch_size) <= 0:
        raise ValueError("Replay capacity, training budget and batch_size must be positive")
    effective = min(configured, total_timesteps)
    if effective < batch_size:
        raise ValueError("CrossQ replay capacity cannot be smaller than batch_size")
    return effective


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
    if algorithm == "ppo":
        rollout_steps = int(cfg["n_steps"])
        minibatch_size = int(cfg["batch_size"])
        if rollout_steps <= 0 or minibatch_size <= 0:
            raise ValueError("PPO rollout and minibatch sizes must be positive.")
        if rollout_steps % minibatch_size != 0:
            raise ValueError(
                "PPO batch_size must divide n_steps so each rollout has full minibatches."
            )
    run_dir = ensure_dir(
        root / benchmark["output_root"] / "runs" / algorithm / scenario / f"seed_{seed}"
    )
    # CrossQ uses a 1024x1024 critic and a gradient update almost every step.
    # Avoid excessive OpenMP thread oversubscription on HPC worker slots.
    # Do not shrink the critic or change the learning update ratio here.
    effective_replay_capacity = None
    if algorithm == "crossq":
        import torch

        thread_count = int(cfg.get("torch_num_threads", 4))
        if thread_count < 1:
            raise ValueError("CrossQ torch_num_threads must be positive")
        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(min(thread_count, 4))
        except RuntimeError:
            # PyTorch permits setting inter-op threads only before any work.
            # The intra-op limit above remains in effect.
            pass
        # A 500k-step run cannot populate a 1m-transition buffer.
        # Keeping the full *effective* history gives identical sampling
        # support without allocating unused replay slots.
        effective_replay_capacity = replay_capacity_for_budget(
            int(cfg["buffer_size"]), int(total_timesteps), int(cfg["batch_size"])
        )

    write_json(run_dir / "system_manifest.json", system_manifest())
    write_json(run_dir / "resolved_config.json", {
        "benchmark": benchmark,
        "algorithm": cfg,
        "runtime": {
            "crossq_effective_replay_capacity": effective_replay_capacity,
        },
    })

    env = build_env(root, scenario, split="train", seed=seed)
    monitor_path = run_dir / "monitor"
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(env, filename=str(monitor_path))

    # The paper trained PPO using VecNormalize. Keep the same fixed input
    # feature scaling for all algorithms *before* PPO's running normalization.
    # Save the statistics so evaluation never recomputes them from test data.
    use_vecnormalize = algorithm == "ppo" and bool(cfg.get("vecnormalize", False))
    if use_vecnormalize:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        env = VecNormalize(
            DummyVecEnv([lambda env=env: env]),
            norm_obs=True,
            norm_reward=True,
            training=True,
        )

    if algorithm == "ppo":
        from stable_baselines3 import PPO

        class ExactBudgetPPO(PPO):
            def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
                # SB3 trains on a full buffer after each collection. Size the
                # final buffer for the remaining steps so it is full at the budget.
                remaining_steps = self._total_timesteps - self.num_timesteps
                remaining_rollout_steps, remainder = divmod(remaining_steps, env.num_envs)
                if remainder:
                    raise ValueError(
                        "PPO training budget must be divisible by the number of environments."
                    )
                steps = min(n_rollout_steps, remaining_rollout_steps)
                if rollout_buffer.buffer_size != steps:
                    rollout_buffer = type(rollout_buffer)(
                        steps,
                        self.observation_space,
                        self.action_space,
                        device=self.device,
                        gamma=self.gamma,
                        gae_lambda=self.gae_lambda,
                        n_envs=env.num_envs,
                        **self.rollout_buffer_kwargs,
                    )
                    self.rollout_buffer = rollout_buffer
                return super().collect_rollouts(env, callback, rollout_buffer, steps)

        model = ExactBudgetPPO(
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
            buffer_size=effective_replay_capacity,
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
        callback = EpisodeCSVCallback.make(
            run_dir / "training_episodes.csv",
            tracker,
            sample_every_steps=int(benchmark.get("compute", {}).get("resource_sample_every_steps", 250)),
        )
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    actual_steps = int(model.num_timesteps)
    if actual_steps != int(total_timesteps):
        env.close()
        raise RuntimeError(
            f"{algorithm} used {actual_steps} environment steps; "
            f"expected exactly {int(total_timesteps)}."
        )
    write_json(
        run_dir / "interaction_budget.json",
        {
            "target_environment_steps": int(total_timesteps),
            "actual_environment_steps": actual_steps,
            "overshoot_steps": actual_steps - int(total_timesteps),
        },
    )
    model.save(run_dir / "model")
    if use_vecnormalize:
        env.save(str(run_dir / "vecnormalize.pkl"))
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

    normalizer = None
    if algorithm == "ppo":
        cfg = load_yaml(root / "configs" / "algorithms" / "ppo.yaml")
        if bool(cfg.get("vecnormalize", False)):
            state_path = run_dir / "vecnormalize.pkl"
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"Missing PPO training normalization statistics: {state_path}. "
                    "Retrain this model with the updated configuration; do not "
                    "reuse an older model with different preprocessing."
                )
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            normalization_env = DummyVecEnv([
                lambda: Monitor(build_env(root, scenario, split=split, seed=seed))
            ])
            normalizer = VecNormalize.load(str(state_path), normalization_env)
            normalizer.training = False  # never leak held-out data into statistics
            normalizer.norm_reward = False

    def policy(obs: np.ndarray, deterministic: bool) -> np.ndarray:
        model_obs = obs
        if normalizer is not None:
            model_obs = normalizer.normalize_obs(obs[None, :])[0]
        action, _ = model.predict(model_obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    try:
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
    finally:
        if normalizer is not None:
            normalizer.close()
    target = run_dir / ("evaluation.csv" if split == "test" else "validation_evaluation.csv")
    save_evaluation(frame, target)
    return target
