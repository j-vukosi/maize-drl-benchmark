from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from maize_benchmark.config import load_benchmark_config, load_yaml
from maize_benchmark.envs.factory import build_env
from maize_benchmark.evaluation import evaluate_policy, save_evaluation
from maize_benchmark.logging_utils import ResourceTracker, ensure_dir, system_manifest, write_json
from maize_benchmark.seeding import seed_everything


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(sizes: list[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        act = activation if index < len(sizes) - 2 else output_activation
        layers.extend([nn.Linear(sizes[index], sizes[index + 1]), act()])
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], low, high):
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden], nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        low_t = torch.as_tensor(low, dtype=torch.float32)
        high_t = torch.as_tensor(high, dtype=torch.float32)
        self.register_buffer("action_scale", (high_t - low_t) / 2.0)
        self.register_buffer("action_bias", (high_t + low_t) / 2.0)

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        features = self.trunk(obs)
        mu = self.mu(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        raw = mu if deterministic else dist.rsample()
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias
        log_prob = None
        if with_logprob:
            log_prob = dist.log_prob(raw)
            correction = torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
            log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)
        return action, log_prob


class DistributionalCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], low, high):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 2], nn.ReLU, nn.Identity)
        low_t = torch.as_tensor(low, dtype=torch.float32)
        high_t = torch.as_tensor(high, dtype=torch.float32)
        self.register_buffer("action_scale", torch.clamp((high_t - low_t) / 2.0, min=1e-6))
        self.register_buffer("action_bias", (high_t + low_t) / 2.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        normalized_action = (action - self.action_bias) / self.action_scale
        out = self.net(torch.cat([obs, normalized_action], dim=-1))
        mean = out[..., :1]
        std = F.softplus(out[..., 1:2]) + 1e-4
        return mean, std


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.max_size = int(size)
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "act": torch.as_tensor(self.actions[idx], device=self.device),
            "rew": torch.as_tensor(self.rewards[idx], device=self.device),
            "obs2": torch.as_tensor(self.next_obs[idx], device=self.device),
            "done": torch.as_tensor(self.dones[idx], device=self.device),
        }


@dataclass
class DSACMetrics:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    q1_mean: float
    q2_mean: float
    std1_mean: float
    std2_mean: float


class DSACTAgent:
    """Compact DSAC-T implementation following the official DSAC-v2 equations.

    Refinements represented here: twin distributional critics, mean-based clipped
    double-Q target selection, target-bound clipping using running critic standard
    deviation, variance-aware loss reweighting, delayed actor updates and automatic
    entropy tuning.
    """

    def __init__(self, obs_dim: int, action_space, cfg: dict[str, Any], device: torch.device):
        self.device = device
        act_dim = int(np.prod(action_space.shape))
        hidden = [int(v) for v in cfg["hidden_sizes"]]
        low, high = action_space.low, action_space.high
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden, low, high).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.q1 = DistributionalCritic(obs_dim, act_dim, hidden, low, high).to(device)
        self.q2 = DistributionalCritic(obs_dim, act_dim, hidden, low, high).to(device)
        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)
        for module in [self.actor_target, self.q1_target, self.q2_target]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["learning_rate_actor"])
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg["learning_rate_critic"])
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg["learning_rate_critic"])
        self.log_alpha = torch.tensor(
            np.log(cfg["initial_alpha"]), dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["learning_rate_alpha"])
        self.target_entropy = -float(act_dim)
        self.gamma = float(cfg["gamma"])
        self.tau = float(cfg["tau"])
        self.tau_b = float(cfg.get("tau_b", self.tau))
        self.policy_delay = int(cfg["policy_delay"])
        self.auto_alpha = bool(cfg["auto_alpha"])
        self.fixed_alpha = float(cfg["initial_alpha"])
        self.huber_delta = float(cfg["huber_delta"])
        self.std_bias = float(cfg["std_bias"])
        self.ratio_min = float(cfg["std_ratio_min"])
        self.ratio_max = float(cfg["std_ratio_max"])
        self.td_bound_multiplier = float(cfg["td_bound_multiplier"])
        self.mean_std1: torch.Tensor | None = None
        self.mean_std2: torch.Tensor | None = None
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return (
            self.log_alpha.exp()
            if self.auto_alpha
            else torch.tensor(self.fixed_alpha, device=self.device)
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(tensor, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _sample_q(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        z = torch.clamp(torch.randn_like(mean), -3.0, 3.0)
        return mean + z * std

    def _target_pair(self, batch: dict[str, torch.Tensor], q_current, running_std):
        with torch.no_grad():
            next_action, next_logp = self.actor_target(batch["obs2"])
            q1_next, s1_next = self.q1_target(batch["obs2"], next_action)
            q2_next, s2_next = self.q2_target(batch["obs2"], next_action)
            q1_sample = self._sample_q(q1_next, s1_next)
            q2_sample = self._sample_q(q2_next, s2_next)
            choose_q1 = q1_next < q2_next
            q_next = torch.minimum(q1_next, q2_next)
            q_next_sample = torch.where(choose_q1, q1_sample, q2_sample)
            target_mean = batch["rew"] + (1 - batch["done"]) * self.gamma * (
                q_next - self.alpha.detach() * next_logp
            )
            target_sample = batch["rew"] + (1 - batch["done"]) * self.gamma * (
                q_next_sample - self.alpha.detach() * next_logp
            )
            bound = self.td_bound_multiplier * running_std
            clipped_difference = torch.clamp(target_sample - q_current, -bound, bound)
            target_bound = q_current + clipped_difference
        return target_mean, target_bound

    def _critic_loss(self, q, std, target_mean, target_bound, running_std):
        std_detached = torch.clamp(std, min=0.0).detach()
        ratio = (running_std.pow(2) / (std_detached.pow(2) + self.std_bias)).clamp(
            self.ratio_min, self.ratio_max
        )
        mean_term = F.huber_loss(q, target_mean, delta=self.huber_delta, reduction="none")
        bound_term = F.huber_loss(
            q.detach(), target_bound, delta=self.huber_delta, reduction="none"
        )
        variance_term = std * (std_detached.pow(2) - bound_term) / (
            std_detached + self.std_bias
        )
        return torch.mean(ratio * (mean_term + variance_term))

    def update(self, batch: dict[str, torch.Tensor]) -> DSACMetrics:
        self.update_count += 1
        q1, std1 = self.q1(batch["obs"], batch["act"])
        q2, std2 = self.q2(batch["obs"], batch["act"])
        batch_std1 = std1.detach().mean()
        batch_std2 = std2.detach().mean()
        self.mean_std1 = (
            batch_std1 if self.mean_std1 is None else (1 - self.tau_b) * self.mean_std1 + self.tau_b * batch_std1
        )
        self.mean_std2 = (
            batch_std2 if self.mean_std2 is None else (1 - self.tau_b) * self.mean_std2 + self.tau_b * batch_std2
        )
        target1, bound1 = self._target_pair(batch, q1.detach(), self.mean_std1.detach())
        target2, bound2 = self._target_pair(batch, q2.detach(), self.mean_std2.detach())
        loss1 = self._critic_loss(q1, std1, target1, bound1, self.mean_std1.detach())
        loss2 = self._critic_loss(q2, std2, target2, bound2, self.mean_std2.detach())
        critic_loss = loss1 + loss2
        self.q1_opt.zero_grad(set_to_none=True)
        self.q2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        actor_loss_value = 0.0
        alpha_loss_value = 0.0
        if self.update_count % self.policy_delay == 0:
            for critic in [self.q1, self.q2]:
                for parameter in critic.parameters():
                    parameter.requires_grad = False
            new_action, logp = self.actor(batch["obs"])
            q1_pi, _ = self.q1(batch["obs"], new_action)
            q2_pi, _ = self.q2(batch["obs"], new_action)
            actor_loss = (self.alpha.detach() * logp - torch.minimum(q1_pi, q2_pi)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.detach())
            for critic in [self.q1, self.q2]:
                for parameter in critic.parameters():
                    parameter.requires_grad = True

            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()
                alpha_loss_value = float(alpha_loss.detach())

            # Match the official DSAC-v2 update schedule: actor, entropy and all
            # target networks advance only on delayed policy-update iterations.
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            self._soft_update(self.actor, self.actor_target)
        return DSACMetrics(
            critic_loss=float(critic_loss.detach()),
            actor_loss=actor_loss_value,
            alpha_loss=alpha_loss_value,
            alpha=float(self.alpha.detach()),
            q1_mean=float(q1.detach().mean()),
            q2_mean=float(q2.detach().mean()),
            std1_mean=float(std1.detach().mean()),
            std2_mean=float(std2.detach().mean()),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.mul_(1 - self.tau).add_(self.tau * source_param)

    def save(self, path: Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "mean_std1": None if self.mean_std1 is None else self.mean_std1.detach().cpu(),
                "mean_std2": None if self.mean_std2 is None else self.mean_std2.detach().cpu(),
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        for name in ["actor", "actor_target", "q1", "q2", "q1_target", "q2_target"]:
            getattr(self, name).load_state_dict(state[name])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.mean_std1 = None if state["mean_std1"] is None else state["mean_std1"].to(self.device)
        self.mean_std2 = None if state["mean_std2"] is None else state["mean_std2"].to(self.device)
        self.update_count = int(state["update_count"])


def _device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def train_dsac_t(
    *, project_root: str | Path, scenario: str, seed: int, total_timesteps: int
) -> Path:
    root = Path(project_root)
    seed_everything(seed)
    benchmark = load_benchmark_config(root)
    cfg = load_yaml(root / "configs" / "algorithms" / "dsac_t.yaml")
    run_dir = ensure_dir(
        root / benchmark["output_root"] / "runs" / "dsac_t" / scenario / f"seed_{seed}"
    )
    write_json(run_dir / "system_manifest.json", system_manifest())
    write_json(run_dir / "resolved_config.json", {"benchmark": benchmark, "algorithm": cfg})
    env = build_env(root, scenario, split="train", seed=seed)
    obs, _ = env.reset(seed=seed)
    device = _device_from_config(benchmark.get("compute", {}).get("device", "auto"))
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = DSACTAgent(obs_dim, env.action_space, cfg, device)
    replay = ReplayBuffer(obs_dim, act_dim, cfg["buffer_size"], device)
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []

    with ResourceTracker() as tracker:
        for step in range(1, int(total_timesteps) + 1):
            if step <= cfg["learning_starts"]:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            if replay.size >= cfg["batch_size"] and step > cfg["learning_starts"]:
                for _ in range(int(cfg["updates_per_step"])):
                    metrics = agent.update(replay.sample(cfg["batch_size"]))
                    if step % 100 == 0:
                        update_rows.append({"timesteps": step, **metrics.__dict__})
            if done:
                episode_rows.append(
                    {
                        "timesteps": step,
                        **{k: info.get(k) for k in [
                            "year", "dry_yield_tonne_per_ha", "total_irrigation_mm",
                            "water_productivity_kg_per_ha_per_mm", "profit",
                            "episode_reward", "stress_exposure", "days"
                        ]},
                    }
                )
                obs, _ = env.reset()
            if step % 1000 == 0:
                tracker.sample()
            if step % int(benchmark["protocol"]["checkpoint_interval"]) == 0:
                agent.save(run_dir / f"checkpoint_{step}.pt")
    agent.save(run_dir / "model.pt")
    write_json(
        run_dir / "interaction_budget.json",
        {
            "target_environment_steps": int(total_timesteps),
            "actual_environment_steps": int(total_timesteps),
            "overshoot_steps": 0,
        },
    )
    pd.DataFrame(episode_rows).to_csv(run_dir / "training_episodes.csv", index=False)
    pd.DataFrame(update_rows).to_csv(run_dir / "training_updates.csv", index=False)
    write_json(run_dir / "compute.json", tracker.as_dict())
    env.close()
    return run_dir


def evaluate_dsac_t(
    *,
    project_root: str | Path,
    scenario: str,
    seed: int,
    episodes_per_year: int,
    split: str = "test",
) -> Path:
    root = Path(project_root)
    benchmark = load_benchmark_config(root)
    cfg = load_yaml(root / "configs" / "algorithms" / "dsac_t.yaml")
    run_dir = root / benchmark["output_root"] / "runs" / "dsac_t" / scenario / f"seed_{seed}"
    probe = build_env(root, scenario, split=split, seed=seed)
    device = _device_from_config(benchmark.get("compute", {}).get("device", "auto"))
    agent = DSACTAgent(
        int(np.prod(probe.observation_space.shape)), probe.action_space, cfg, device
    )
    probe.close()
    agent.load(run_dir / "model.pt")

    def policy(obs: np.ndarray, deterministic: bool) -> np.ndarray:
        return agent.act(obs, deterministic=deterministic)

    frame = evaluate_policy(
        project_root=root,
        algorithm="dsac_t",
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
