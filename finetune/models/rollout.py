from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .checkpoint import autocast_context, resolve_amp_settings


@dataclass
class RolloutBatch:
    states: torch.Tensor
    labels: torch.Tensor
    transition_times: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor | None = None
    reward_classifier: torch.Tensor | None = None
    reward_musiq: torch.Tensor | None = None
    advantages: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[1] - 1)

    @property
    def terminal_states(self) -> torch.Tensor:
        return self.states[:, -1]


def compute_sigma(t: torch.Tensor, noise_level: float, t_eps: float) -> torch.Tensor:
    numerator = torch.clamp(t, min=t_eps)
    denominator = torch.clamp(1.0 - t, min=t_eps)
    return float(noise_level) * torch.sqrt(numerator / denominator)


def select_training_step_indices(total_steps: int, train_steps: int) -> torch.Tensor:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive.")
    if train_steps >= total_steps:
        return torch.arange(total_steps, dtype=torch.long)
    linspace = torch.linspace(0, total_steps - 1, steps=train_steps)
    indices = torch.unique(linspace.round().long(), sorted=True)
    if indices.numel() == train_steps:
        return indices
    all_indices = torch.arange(total_steps, dtype=torch.long)
    missing = all_indices[~torch.isin(all_indices, indices)]
    combined = torch.cat([indices, missing[: train_steps - indices.numel()]])
    return torch.sort(combined).values


def gaussian_log_prob(target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    variance = std.square()
    dims = target[0].numel()
    log_normalizer = 0.5 * dims * math.log(2.0 * math.pi) + dims * torch.log(std)
    quadratic = 0.5 * (target - mean).square().flatten(1).sum(dim=1) / variance
    return -(quadratic + log_normalizer)


def transition_mean(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    dt: float,
    noise_level: float,
    t_eps: float,
    amp_settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = compute_sigma(t, noise_level=noise_level, t_eps=t_eps)
    sigma_sq = sigma.square().view(-1, 1, 1, 1)
    t_view = torch.clamp(t, min=t_eps).view(-1, 1, 1, 1)
    with autocast_context(amp_settings):
        velocity = model(x_t, t, labels)
    drift = velocity.float() + (sigma_sq / (2.0 * t_view)) * (x_t + (1.0 - t_view) * velocity.float())
    mean = x_t + float(dt) * drift
    return mean, sigma, velocity.float(), drift


def compute_transition_logprob(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    dt: float,
    noise_level: float,
    t_eps: float,
    amp_settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, sigma, velocity, drift = transition_mean(
        model=model,
        x_t=x_t,
        t=t,
        labels=labels,
        dt=dt,
        noise_level=noise_level,
        t_eps=t_eps,
        amp_settings=amp_settings,
    )
    step_std = torch.clamp(sigma * math.sqrt(dt), min=1e-6)
    log_prob = gaussian_log_prob(x_next, mean, step_std)
    return log_prob, mean, step_std, velocity, drift


@torch.no_grad()
def rollout_sde_with_logprobs(
    model: torch.nn.Module,
    labels: torch.Tensor,
    num_steps: int,
    noise_level: float,
    device: torch.device,
    image_shape: tuple[int, int, int] = (3, 64, 64),
    t_eps: float = 1e-3,
    amp_settings=None,
) -> RolloutBatch:
    if noise_level <= 0.0:
        raise ValueError("noise_level must be positive for Flow-GRPO rollouts.")
    if amp_settings is None:
        amp_settings = resolve_amp_settings(device, enabled=device.type == "cuda")

    batch_size = int(labels.shape[0])
    dt = 1.0 / float(num_steps)
    states = torch.empty(batch_size, num_steps + 1, *image_shape, device=device, dtype=torch.float32)
    log_probs = torch.empty(batch_size, num_steps, device=device, dtype=torch.float32)
    transition_times = torch.empty(num_steps, device=device, dtype=torch.float32)
    states[:, 0] = torch.randn(batch_size, *image_shape, device=device)

    x_t = states[:, 0]
    for step in range(num_steps):
        t_value = step * dt
        t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)
        transition_times[step] = t_value
        log_prob, mean, step_std, _, _ = compute_transition_logprob(
            model=model,
            x_t=x_t,
            x_next=x_t,
            t=t,
            labels=labels,
            dt=dt,
            noise_level=noise_level,
            t_eps=t_eps,
            amp_settings=amp_settings,
        )
        noise = torch.randn_like(x_t)
        x_next = mean + step_std.view(-1, 1, 1, 1) * noise
        log_probs[:, step] = gaussian_log_prob(x_next, mean, step_std)
        states[:, step + 1] = x_next
        x_t = x_next
    return RolloutBatch(states=states, labels=labels, transition_times=transition_times, log_probs=log_probs)


@torch.no_grad()
def deterministic_euler_sample(
    model: torch.nn.Module,
    labels: torch.Tensor,
    num_steps: int,
    device: torch.device,
    image_shape: tuple[int, int, int] = (3, 64, 64),
    amp_settings=None,
) -> torch.Tensor:
    if amp_settings is None:
        amp_settings = resolve_amp_settings(device, enabled=device.type == "cuda")
    batch_size = int(labels.shape[0])
    dt = 1.0 / float(num_steps)
    x = torch.randn(batch_size, *image_shape, device=device, dtype=torch.float32)
    for step in range(num_steps):
        t = torch.full((batch_size,), step * dt, device=device, dtype=torch.float32)
        with autocast_context(amp_settings):
            velocity = model(x, t, labels)
        x = x + dt * velocity.float()
    return x
