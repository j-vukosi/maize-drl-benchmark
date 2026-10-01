from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maize_benchmark.config import load_benchmark_config, load_yaml
from maize_benchmark.envs.factory import build_env
from maize_benchmark.evaluation import evaluate_policy, save_evaluation
from maize_benchmark.logging_utils import (
    ResourceTracker,
    ensure_dir,
    system_manifest,
    write_json,
)
from maize_benchmark.seeding import seed_everything


# Ray 2.49 DreamerV3 uses PyTorch. TensorFlow is not part of the benchmark
# runtime, but some optional RLlib imports may still probe TensorFlow when it
# remains installed in the virtual environment.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

EXPECTED_RAY_VERSION = "2.49.0"


class _ExpectedThirdPartyLogFilter(logging.Filter):
    """Hide narrowly identified upstream notices.

    Gymnasium dtype, shape and containment warnings are deliberately not
    filtered because they indicate an invalid benchmark environment contract.
    """

    _ignored_fragments = (
        # Optional TensorFlow imports made internally by RLlib.
        "The name tf.logging.set_verbosity is deprecated",
        "The name tf.losses.sparse_softmax_cross_entropy is deprecated",
        "The name tf.logging.TaskLevelStatusMessage is deprecated",
        "The name tf.control_flow_v2_enabled is deprecated",

        # Expected Ray 2.49 DreamerV3 notices.
        "You are running DreamerV3 on the new API stack",
        "`RLModule(config=[RLModuleConfig object])` has been deprecated",

        # Optional system-monitoring notice.
        "Install gputil for GPU system monitoring",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()

        return not any(
            fragment in message
            for fragment in self._ignored_fragments
        )


_EXPECTED_LOG_FILTER = _ExpectedThirdPartyLogFilter()


def _install_expected_warning_filters() -> None:
    """Install narrow warning and log filters for pinned upstream libraries."""

    # Ray emits these through Python's warnings framework.
    warning_patterns = (
        r".*`UnifiedLogger` will be removed.*",
        r".*The `JsonLogger interface is deprecated.*",
        r".*The `CSVLogger interface is deprecated.*",
        r".*The `TBXLogger interface is deprecated.*",
    )

    for pattern in warning_patterns:
        warnings.filterwarnings(
            "ignore",
            message=pattern,
            category=Warning,
        )

    # Logger filters must be attached before Ray/RLlib is imported so optional
    # TensorFlow compatibility notices are filtered during import.
    logger_names = (
        "",
        "tensorflow",
        "ray",
        "ray.rllib",
        "ray.rllib.algorithms.algorithm_config",
        "ray.rllib.utils.deprecation",
        "ray.tune",
        "ray.tune.utils.util",
    )

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.addFilter(_EXPECTED_LOG_FILTER)

        for handler in logger.handlers:
            handler.addFilter(_EXPECTED_LOG_FILTER)

    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        handler.addFilter(_EXPECTED_LOG_FILTER)


# Install before any function imports Ray or RLlib.
_install_expected_warning_filters()


def _as_numeric_scalar(value: Any) -> float | None:
    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return float(value)

    item = getattr(value, "item", None)

    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError):
            return None

        if isinstance(
            scalar,
            (int, float, np.integer, np.floating),
        ):
            return float(scalar)

    return None


def _find_numeric(
    payload: Any,
    preferred_keys: tuple[str, ...],
) -> float | None:
    if not isinstance(payload, dict):
        return None

    for key in preferred_keys:
        if key not in payload:
            continue

        scalar = _as_numeric_scalar(payload[key])

        if scalar is not None:
            return scalar

    for value in payload.values():
        if isinstance(value, dict):
            found = _find_numeric(
                value,
                preferred_keys,
            )

            if found is not None:
                return found

    return None


def _extract_training_row(
    result: dict[str, Any],
) -> dict[str, Any]:
    environment_steps = _find_numeric(
        result,
        (
            "num_env_steps_sampled_lifetime",
            "num_env_steps_sampled",
            "timesteps_total",
            "agent_timesteps_total",
        ),
    )

    return {
        "training_iteration": int(
            result.get("training_iteration", 0)
        ),
        "environment_steps": int(
            environment_steps or 0
        ),
        "episode_return_mean": _find_numeric(
            result,
            (
                "episode_return_mean",
                "episode_reward_mean",
            ),
        ),
        "episode_len_mean": _find_numeric(
            result,
            ("episode_len_mean",),
        ),
        "learner_loss": _find_numeric(
            result,
            (
                "WORLD_MODEL_L_total",
                "world_model_loss",
                "total_loss",
            ),
        ),
        "time_total_s": float(
            result.get("time_total_s", 0.0)
        ),
    }


def _assert_runtime_versions(
    ray_module: Any,
    torch_module: Any,
) -> None:
    ray_version = str(ray_module.__version__)

    if ray_version != EXPECTED_RAY_VERSION:
        raise RuntimeError(
            "This DreamerV3 runner targets "
            f"Ray {EXPECTED_RAY_VERSION}, but found "
            f"Ray {ray_version}. Activate .venv_dreamer and run:\n\n"
            "python -m pip install --upgrade "
            f'"ray[rllib]=={EXPECTED_RAY_VERSION}"'
        )

    if not hasattr(torch_module, "__version__"):
        raise RuntimeError(
            "PyTorch is unavailable in the Dreamer environment."
        )


def _validate_environment_contract(env) -> None:
    """Validate the final environment exposed to DreamerV3."""

    observation_space = getattr(
        env,
        "observation_space",
        None,
    )
    action_space = getattr(
        env,
        "action_space",
        None,
    )

    if observation_space is None:
        raise TypeError(
            "build_env returned an environment without "
            "an observation_space."
        )

    if action_space is None:
        raise TypeError(
            "build_env returned an environment without "
            "an action_space."
        )

    expected_dtype = np.dtype(np.float32)

    observation_dtype = np.dtype(
        observation_space.dtype
    )

    if observation_dtype != expected_dtype:
        raise TypeError(
            "DreamerV3 requires a float32 observation space, "
            f"but build_env returned {observation_dtype}."
        )

    observation_low = np.asarray(
        observation_space.low
    )
    observation_high = np.asarray(
        observation_space.high
    )

    if observation_low.dtype != expected_dtype:
        raise TypeError(
            "Observation-space lower bounds must use float32, "
            f"but received {observation_low.dtype}."
        )

    if observation_high.dtype != expected_dtype:
        raise TypeError(
            "Observation-space upper bounds must use float32, "
            f"but received {observation_high.dtype}."
        )

    if observation_low.shape != observation_high.shape:
        raise ValueError(
            "Observation-space bounds have different shapes: "
            f"low={observation_low.shape}, "
            f"high={observation_high.shape}."
        )

    if observation_space.shape != observation_low.shape:
        raise ValueError(
            "Observation-space declared shape does not match "
            "its bound arrays: "
            f"space={observation_space.shape}, "
            f"bounds={observation_low.shape}."
        )

    action_dtype = getattr(
        action_space,
        "dtype",
        None,
    )

    if action_dtype is not None:
        action_dtype = np.dtype(action_dtype)

        if action_dtype != expected_dtype:
            raise TypeError(
                "DreamerV3 requires a float32 action space, "
                f"but build_env returned {action_dtype}."
            )

        action_low = np.asarray(
            action_space.low
        )
        action_high = np.asarray(
            action_space.high
        )

        if action_low.dtype != expected_dtype:
            raise TypeError(
                "Action-space lower bounds must use float32, "
                f"but received {action_low.dtype}."
            )

        if action_high.dtype != expected_dtype:
            raise TypeError(
                "Action-space upper bounds must use float32, "
                f"but received {action_high.dtype}."
            )


def _register_env(
    root: Path,
    scenario: str,
    split: str,
    seed: int,
) -> str:
    from ray.tune.registry import register_env

    env_name = (
        f"maize_benchmark_{scenario}_{split}_{seed}"
    )

    def creator(env_config):
        context_seed = int(
            env_config.get("seed", seed)
        )
        fixed_year = env_config.get(
            "fixed_year"
        )

        env = build_env(
            root,
            scenario,
            split=split,
            seed=context_seed,
            fixed_year=(
                None
                if fixed_year is None
                else int(fixed_year)
            ),
        )

        try:
            _validate_environment_contract(env)
        except Exception:
            env.close()
            raise

        return env

    register_env(
        env_name,
        creator,
    )

    return env_name


def _build_algorithm(config):
    build_algo = getattr(
        config,
        "build_algo",
        None,
    )

    if callable(build_algo):
        return build_algo()

    return config.build()


def _save_checkpoint(
    algo,
    target: Path,
) -> str:
    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_to_path = getattr(
        algo,
        "save_to_path",
        None,
    )

    if callable(save_to_path):
        returned = save_to_path(
            str(target)
        )
        return str(returned or target)

    result = algo.save(
        str(target)
    )

    if (
        hasattr(result, "checkpoint")
        and hasattr(result.checkpoint, "path")
    ):
        return str(
            result.checkpoint.path
        )

    return str(result)


def train_dreamerv3(
    *,
    project_root: str | Path,
    scenario: str,
    seed: int,
    total_timesteps: int,
) -> Path:
    # Filters must be active before importing Ray/RLlib.
    _install_expected_warning_filters()

    try:
        import ray
        import torch

        from ray.rllib.algorithms.dreamerv3.dreamerv3 import (
            DreamerV3Config,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DreamerV3 dependencies are unavailable. "
            "Activate or use the .venv_dreamer environment "
            "before running DreamerV3."
        ) from exc

    _install_expected_warning_filters()
    _assert_runtime_versions(
        ray,
        torch,
    )

    if int(total_timesteps) <= 0:
        raise ValueError(
            "total_timesteps must be greater than zero."
        )

    root = Path(
        project_root
    ).resolve()

    seed_everything(
        seed,
        deterministic_torch=False,
    )

    benchmark = load_benchmark_config(
        root
    )

    cfg = load_yaml(
        root
        / "configs"
        / "algorithms"
        / "dreamerv3.yaml"
    )

    run_dir = ensure_dir(
        root
        / benchmark["output_root"]
        / "runs"
        / "dreamerv3"
        / scenario
        / f"seed_{seed}"
    )

    write_json(
        run_dir
        / "system_manifest.json",
        system_manifest(),
    )

    write_json(
        run_dir
        / "resolved_config.json",
        {
            "benchmark": benchmark,
            "algorithm": cfg,
            "runtime": {
                "ray": ray.__version__,
                "torch": torch.__version__,
                "framework": "torch",
                "cuda_available": bool(
                    torch.cuda.is_available()
                ),
                "cuda_device_count": int(
                    torch.cuda.device_count()
                ),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        },
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
    )

    # Ray may add handlers during initialization.
    _install_expected_warning_filters()

    env_name = _register_env(
        root,
        scenario,
        "train",
        seed,
    )

    dreamer_config = (
        DreamerV3Config()
        .framework("torch")
        .environment(
            env=env_name,
            env_config={
                "seed": seed,
            },
        )
        .training(
            model_size=cfg["model_size"],
            training_ratio=cfg["training_ratio"],
            batch_size_B=cfg["batch_size_B"],
            batch_length_T=cfg["batch_length_T"],
            horizon_H=cfg["horizon_H"],
            gamma=cfg["gamma"],
        )
        .reporting(
            metrics_num_episodes_for_smoothing=1,
            min_sample_timesteps_per_iteration=64,
            report_images_and_videos=False,
            report_dream_data=False,
            report_individual_batch_item_stats=False,
        )
    )

    if hasattr(
        dreamer_config,
        "env_runners",
    ):
        dreamer_config = (
            dreamer_config.env_runners(
                num_env_runners=int(
                    cfg.get(
                        "num_env_runners",
                        0,
                    )
                ),
                num_envs_per_env_runner=int(
                    cfg.get(
                        "num_envs_per_env_runner",
                        1,
                    )
                ),
            )
        )

    # Ray 2.49 DreamerV3 uses the Learner API. When num_learners=0,
    # the learner runs locally. Explicitly assign the allocated GPU to
    # that local learner when CUDA is available. Without this setting,
    # Ray defaults to zero GPUs for the learner even when the SLURM job
    # has an NVIDIA GPU allocated.
    if hasattr(
        dreamer_config,
        "learners",
    ):
        dreamer_config = (
            dreamer_config.learners(
                num_learners=int(
                    cfg.get(
                        "num_learners",
                        0,
                    )
                ),
                num_gpus_per_learner=(
                    1
                    if torch.cuda.is_available()
                    else 0
                ),
            )
        )

    if hasattr(
        dreamer_config,
        "debugging",
    ):
        dreamer_config = (
            dreamer_config.debugging(
                seed=seed
            )
        )

    dreamer_config = dreamer_config.env_runners(

        rollout_fragment_length=int(

            cfg.get("rollout_fragment_length", 1)

        )

    )



    rows: list[dict[str, Any]] = []

    checkpoint_interval = int(
        cfg.get(
            "checkpoint_interval_steps",
            50_000,
        )
    )

    if checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval_steps must be positive."
        )

    next_checkpoint = checkpoint_interval

    checkpoint_records: list[
        dict[str, Any]
    ] = []

    algo = None
    tracker = ResourceTracker()
    env_steps = 0

    try:
        algo = _build_algorithm(
            dreamer_config
        )

        stagnant_iterations = 0

        with tracker:
            while env_steps < int(
                total_timesteps
            ):
                result = algo.train()
                row = _extract_training_row(
                    result
                )

                reported_steps = int(
                    row["environment_steps"]
                )

                if reported_steps > env_steps:
                    env_steps = reported_steps
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1

                rows.append(row)
                tracker.sample()

                print(
                    "DreamerV3 "
                    f"scenario={scenario} "
                    f"seed={seed} "
                    f"iteration="
                    f"{row['training_iteration']} "
                    f"env_steps={env_steps}"
                )

                if stagnant_iterations >= 5:
                    write_json(
                        run_dir
                        / "dreamer_metric_diagnostic.json",
                        {
                            "message": (
                                "RLlib did not expose an "
                                "increasing environment-step "
                                "metric."
                            ),
                            "last_result_top_level_keys": (
                                sorted(
                                    result.keys()
                                )
                            ),
                            "last_extracted_row": row,
                        },
                    )

                    raise RuntimeError(
                        "DreamerV3 training made no "
                        "observable environment-step "
                        "progress for five iterations. "
                        "See "
                        "dreamer_metric_diagnostic.json."
                    )

                while (
                    env_steps
                    >= next_checkpoint
                ):
                    checkpoint_target = (
                        run_dir
                        / "checkpoints"
                        / f"step_{env_steps}"
                    )

                    location = _save_checkpoint(
                        algo,
                        checkpoint_target,
                    )

                    checkpoint_records.append(
                        {
                            "environment_steps": (
                                env_steps
                            ),
                            "path": location,
                        }
                    )

                    next_checkpoint += (
                        checkpoint_interval
                    )

        final_checkpoint = (
            _save_checkpoint(
                algo,
                run_dir
                / "checkpoint_final",
            )
        )

        checkpoint_records.append(
            {
                "environment_steps": (
                    env_steps
                ),
                "path": final_checkpoint,
            }
        )

    finally:
        if algo is not None:
            try:
                algo.stop()
            except Exception as stop_error:
                print(
                    "Warning: DreamerV3 cleanup "
                    f"reported: {stop_error}"
                )

        ray.shutdown()

    pd.DataFrame(
        rows
    ).to_csv(
        run_dir
        / "training_iterations.csv",
        index=False,
    )

    write_json(
        run_dir
        / "checkpoints.json",
        checkpoint_records,
    )

    write_json(
        run_dir
        / "interaction_budget.json",
        {
            "target_environment_steps": int(
                total_timesteps
            ),
            "actual_environment_steps": int(
                env_steps
            ),
            "overshoot_steps": (
                int(env_steps)
                - int(total_timesteps)
            ),
        },
    )

    write_json(
        run_dir
        / "compute.json",
        tracker.as_dict(),
    )

    return run_dir


class RllibDreamerPolicy:
    """State-preserving PyTorch adapter for DreamerV3 inference.

    Ray 2.49 DreamerV3 uses an RLModule instead of the older
    Policy API. Observations contain batch and time dimensions,
    and recurrent state remains batched between calls.
    """

    def __init__(self, algo):
        _install_expected_warning_filters()

        try:
            import torch
            import tree

            from ray.rllib.core import (
                DEFAULT_MODULE_ID,
            )
            from ray.rllib.core.columns import (
                Columns,
            )
        except ImportError as exc:
            raise RuntimeError(
                "RLlib DreamerV3 inference requires "
                "PyTorch, dm-tree and Ray 2.49.0."
            ) from exc

        self.algo = algo
        self.torch = torch
        self.tree = tree
        self.Columns = Columns

        self.module = self._resolve_module(
            algo,
            DEFAULT_MODULE_ID,
        )

        if hasattr(
            self.module,
            "eval",
        ):
            self.module.eval()

        self.device = self._resolve_device(
            self.module
        )

        self.action_space = getattr(
            self.module,
            "action_space",
            None,
        )

        if self.action_space is None:
            env_runner = getattr(
                algo,
                "env_runner",
                None,
            )
            vector_env = getattr(
                env_runner,
                "env",
                None,
            )
            self.action_space = getattr(
                vector_env,
                "single_action_space",
                None,
            )

        self.state: Any = None
        self.episode_start = True

        self.reset_episode()

    @staticmethod
    def _resolve_module(
        algo,
        default_module_id,
    ):
        env_runner = getattr(
            algo,
            "env_runner",
            None,
        )
        module = getattr(
            env_runner,
            "module",
            None,
        )

        if (
            module is not None
            and hasattr(
                module,
                "forward_inference",
            )
        ):
            return module

        get_module = getattr(
            algo,
            "get_module",
            None,
        )

        if callable(get_module):
            calls = (
                lambda: get_module(
                    default_module_id
                ),
                lambda: get_module(
                    module_id=(
                        default_module_id
                    )
                ),
                lambda: get_module(),
            )

            for call in calls:
                try:
                    candidate = call()
                except (
                    TypeError,
                    KeyError,
                    AttributeError,
                ):
                    continue

                if hasattr(
                    candidate,
                    "forward_inference",
                ):
                    return candidate

                try:
                    nested = candidate[
                        default_module_id
                    ]
                except (
                    TypeError,
                    KeyError,
                ):
                    nested = None

                if (
                    nested is not None
                    and hasattr(
                        nested,
                        "forward_inference",
                    )
                ):
                    return nested

        learner_group = getattr(
            algo,
            "learner_group",
            None,
        )
        learner = getattr(
            learner_group,
            "_learner",
            None,
        )
        multi_module = getattr(
            learner,
            "module",
            None,
        )

        if multi_module is not None:
            try:
                candidate = multi_module[
                    default_module_id
                ]
            except (
                TypeError,
                KeyError,
            ):
                candidate = multi_module

            if hasattr(
                candidate,
                "forward_inference",
            ):
                return candidate

        raise RuntimeError(
            "Could not locate the restored "
            "DreamerV3 PyTorch RLModule."
        )

    @staticmethod
    def _resolve_device(module):
        try:
            parameter = next(
                module.parameters()
            )
        except (
            StopIteration,
            AttributeError,
        ):
            return "cpu"

        return parameter.device

    def _prepare_initial_state_tensor(
        self,
        value,
    ):
        if self.torch.is_tensor(value):
            tensor = value.to(
                self.device
            )
        else:
            tensor = self.torch.as_tensor(
                value,
                device=self.device,
            )

        # RLlib returns initial states without
        # the batch dimension.
        return tensor.unsqueeze(0)

    def _detach_state_tensor(
        self,
        value,
    ):
        if self.torch.is_tensor(value):
            return value.detach()

        return value

    def reset_episode(self) -> None:
        initial_state = (
            self.module.get_initial_state()
        )

        self.state = (
            self.tree.map_structure(
                self._prepare_initial_state_tensor,
                initial_state,
            )
        )

        self.episode_start = True

    def __call__(
        self,
        obs: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        observation = np.asarray(
            obs,
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(observation)
        ):
            raise ValueError(
                "DreamerV3 evaluation received "
                "a non-finite observation."
            )

        observation_tensor = (
            self.torch.as_tensor(
                observation,
                dtype=self.torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        is_first_tensor = (
            self.torch.tensor(
                [
                    1.0
                    if self.episode_start
                    else 0.0
                ],
                dtype=self.torch.float32,
                device=self.device,
            )
        )

        batch = {
            self.Columns.OBS: (
                observation_tensor
            ),
            self.Columns.STATE_IN: (
                self.state
            ),
            "is_first": is_first_tensor,
        }

        forward = (
            self.module.forward_inference
            if deterministic
            else self.module.forward_exploration
        )

        with self.torch.no_grad():
            output = forward(batch)

        self.state = (
            self.tree.map_structure(
                self._detach_state_tensor,
                output[
                    self.Columns.STATE_OUT
                ],
            )
        )

        actions_tensor = output[
            self.Columns.ACTIONS
        ]

        actions = (
            actions_tensor
            .detach()
            .cpu()
            .numpy()
        )

        # [B, T, action] -> [action].
        while actions.ndim > 1:
            actions = actions[0]

        action = np.asarray(
            actions,
            dtype=np.float32,
        ).reshape(-1)

        action = np.clip(
            action,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

        if self.action_space is not None:
            low = np.asarray(
                self.action_space.low,
                dtype=np.float32,
            )
            high = np.asarray(
                self.action_space.high,
                dtype=np.float32,
            )

            action = np.clip(
                action,
                low,
                high,
            ).astype(
                np.float32
            )

        self.episode_start = False

        return action


def _read_final_checkpoint(
    run_dir: Path,
) -> str:
    manifest = (
        run_dir
        / "checkpoints.json"
    )

    if not manifest.exists():
        default = (
            run_dir
            / "checkpoint_final"
        )

        if default.exists():
            return str(default)

        raise FileNotFoundError(
            "No DreamerV3 checkpoint found "
            f"under {run_dir}."
        )

    rows = json.loads(
        manifest.read_text(
            encoding="utf-8"
        )
    )

    if not rows:
        raise RuntimeError(
            "Checkpoint manifest is empty: "
            f"{manifest}"
        )

    return str(
        rows[-1]["path"]
    )


def evaluate_dreamerv3(
    *,
    project_root: str | Path,
    scenario: str,
    seed: int,
    episodes_per_year: int,
    split: str = "test",
) -> Path:
    _install_expected_warning_filters()

    try:
        import ray
        import torch

        from ray.rllib.algorithms.algorithm import (
            Algorithm,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Activate .venv_dreamer before "
            "DreamerV3 evaluation."
        ) from exc

    _install_expected_warning_filters()
    _assert_runtime_versions(
        ray,
        torch,
    )

    root = Path(
        project_root
    ).resolve()

    benchmark = load_benchmark_config(
        root
    )

    run_dir = (
        root
        / benchmark["output_root"]
        / "runs"
        / "dreamerv3"
        / scenario
        / f"seed_{seed}"
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
    )

    _install_expected_warning_filters()

    # The restored checkpoint refers to the registered training
    # environment name.
    _register_env(
        root,
        scenario,
        "train",
        seed,
    )

    checkpoint = (
        _read_final_checkpoint(
            run_dir
        )
    )

    algo = None

    try:
        algo = (
            Algorithm.from_checkpoint(
                checkpoint
            )
        )

        policy = RllibDreamerPolicy(
            algo
        )

        frame = evaluate_policy(
            project_root=root,
            algorithm="dreamerv3",
            scenario=scenario,
            seed=seed,
            policy=policy,
            episodes_per_year=(
                episodes_per_year
            ),
            deterministic=True,
            split=split,
        )

    finally:
        if algo is not None:
            try:
                algo.stop()
            except Exception as stop_error:
                print(
                    "Warning: DreamerV3 "
                    "evaluation cleanup reported: "
                    f"{stop_error}"
                )

        ray.shutdown()

    target = run_dir / (
        "evaluation.csv"
        if split == "test"
        else "validation_evaluation.csv"
    )

    save_evaluation(
        frame,
        target,
    )

    return target