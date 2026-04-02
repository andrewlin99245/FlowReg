#!/usr/bin/env python3
"""
Toy 2D experiment: CFM pretraining → SDE-based RL finetuning → ODE evaluation.

Compares three models:
  1. Base (pretrained CFM only)
  2. RL-only (finetuned with terminal reward, no regularization)
  3. RL+GCR (finetuned with terminal reward + Geometric Consistency Regularization)

The experiment demonstrates that GCR preserves coherent transport structure
during RL finetuning while still achieving strong terminal reward.

Pipeline:
  - Source distribution: 2D annulus
  - Target distribution: 8 isotropic Gaussians on a ring
  - Pretraining: Conditional Flow Matching (CFM) with linear interpolation paths
  - RL finetuning: pathwise differentiable reward through SDE (Euler-Maruyama) rollouts
  - Evaluation: deterministic ODE (Euler) rollouts
  - GCR: penalizes deviation from the bridge velocity v_bridge(x_t, x_1, t) = (x_1 - x_t)/(1 - t)
"""

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Seed
    seed: int = 42

    # Source distribution (annulus)
    r_min: float = 0.6
    r_max: float = 1.0

    # Target distribution (K Gaussians on a ring)
    num_modes: int = 8
    target_radius: float = 2.0
    target_std: float = 0.08

    # Rewarded modes (0-indexed)
    rewarded_modes: List[int] = field(default_factory=lambda: [1, 3, 6])
    reward_sigma: float = 0.15

    # Model architecture
    hidden_dim: int = 128
    num_hidden_layers: int = 3

    # Pretraining (CFM)
    pretrain_steps: int = 10000
    pretrain_batch_size: int = 512
    pretrain_lr: float = 1e-3

    # RL finetuning
    finetune_steps: int = 3000
    finetune_batch_size: int = 256
    finetune_lr: float = 3e-4

    # Rollout settings
    num_rollout_steps: int = 100
    sde_noise_scale: float = 0.1

    # GCR
    lambda_gcr: float = 5.0
    gcr_t_max: float = 0.98  # exclude steps too close to t=1 for numerical stability

    # Plotting
    num_plot_trajectories: int = 200
    num_eval_samples: int = 1024

    # Output
    output_dir: str = "outputs/toy_gcr_experiment"

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def config_from_args() -> Config:
    """Parse command-line arguments into a Config."""
    parser = argparse.ArgumentParser(description="Toy GCR experiment")
    cfg = Config()
    for name, val in vars(cfg).items():
        if name == "rewarded_modes":
            parser.add_argument(f"--{name}", type=int, nargs="+", default=val)
        elif isinstance(val, bool):
            parser.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif isinstance(val, (int, float, str)):
            parser.add_argument(f"--{name}", type=type(val), default=val)
    args = parser.parse_args()
    for name in vars(cfg):
        if hasattr(args, name):
            setattr(cfg, name, getattr(args, name))
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Distributions
# ═══════════════════════════════════════════════════════════════════════════════

def sample_annulus(n: int, r_min: float, r_max: float, device: torch.device) -> torch.Tensor:
    """Sample n points approximately uniformly from a 2D annulus.

    Uses inverse CDF sampling for the radius so density is uniform in area.
    """
    theta = torch.rand(n, device=device) * 2 * math.pi
    # For uniform area sampling: r = sqrt(U * (r_max^2 - r_min^2) + r_min^2)
    u = torch.rand(n, device=device)
    r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def get_mode_centers(num_modes: int, radius: float, device: torch.device) -> torch.Tensor:
    """Return (num_modes, 2) tensor of Gaussian mode centers on a ring."""
    angles = torch.linspace(0, 2 * math.pi, num_modes + 1, device=device)[:-1]
    cx = radius * torch.cos(angles)
    cy = radius * torch.sin(angles)
    return torch.stack([cx, cy], dim=-1)


def sample_target(n: int, centers: torch.Tensor, std: float) -> torch.Tensor:
    """Sample from a mixture of isotropic Gaussians (equal weights)."""
    K = centers.shape[0]
    indices = torch.randint(0, K, (n,), device=centers.device)
    samples = centers[indices] + std * torch.randn(n, 2, device=centers.device)
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Reward
# ═══════════════════════════════════════════════════════════════════════════════

def terminal_reward(x: torch.Tensor, centers: torch.Tensor,
                    rewarded_indices: List[int], sigma: float) -> torch.Tensor:
    """Compute terminal reward: max over rewarded modes of exp(-||x - c_k||^2 / (2 sigma^2)).

    Args:
        x: (batch, 2) terminal states
        centers: (K, 2) all mode centers
        rewarded_indices: list of rewarded mode indices
        sigma: reward bandwidth

    Returns:
        (batch,) reward values
    """
    rewarded_centers = centers[rewarded_indices]  # (R, 2)
    # (batch, 1, 2) - (1, R, 2) -> (batch, R)
    dists_sq = ((x.unsqueeze(1) - rewarded_centers.unsqueeze(0)) ** 2).sum(-1)
    per_mode = torch.exp(-dists_sq / (2 * sigma**2))
    return per_mode.max(dim=1).values


def assigned_terminal_reward(x: torch.Tensor, x0: torch.Tensor,
                             centers: torch.Tensor,
                             rewarded_indices: List[int],
                             sigma: float) -> torch.Tensor:
    """Reward each terminal point based on its nearest rewarded mode (assigned by source angle).

    Each source point x0 is assigned to the rewarded mode whose angular position
    is closest to x0's angle on the annulus. The reward then measures how close
    the terminal point x lands to its *assigned* mode, rather than the best mode.

    This prevents mode collapse: different parts of the annulus are steered toward
    different rewarded modes, preserving multi-modal transport structure.
    """
    rewarded_centers = centers[rewarded_indices]  # (R, 2)
    # Assign each source point to closest rewarded mode by angular distance
    x0_angle = torch.atan2(x0[:, 1], x0[:, 0])  # (batch,)
    rc_angle = torch.atan2(rewarded_centers[:, 1], rewarded_centers[:, 0])  # (R,)
    # Angular distance (handle wraparound)
    angle_diff = x0_angle.unsqueeze(1) - rc_angle.unsqueeze(0)  # (batch, R)
    angle_diff = torch.remainder(angle_diff + math.pi, 2 * math.pi) - math.pi
    assignments = angle_diff.abs().argmin(dim=1)  # (batch,) index into rewarded_centers

    assigned_centers = rewarded_centers[assignments]  # (batch, 2)
    dists_sq = ((x - assigned_centers) ** 2).sum(-1)
    return torch.exp(-dists_sq / (2 * sigma**2))


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """Lightweight sinusoidal embedding for scalar time t in [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) or (batch, 1)
        t = t.view(-1, 1)  # (batch, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t * freqs.unsqueeze(0)  # (batch, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, dim)


class VelocityModel(nn.Module):
    """MLP velocity field u_theta(x, t) -> R^2.

    Input: 2D state x concatenated with a sinusoidal time embedding of t.
    Output: 2D velocity.
    """
    def __init__(self, hidden_dim: int = 128, num_hidden_layers: int = 3, time_embed_dim: int = 16):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        input_dim = 2 + time_embed_dim

        layers = []
        in_d = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.SiLU())
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2) spatial state
            t: (batch,) or (batch, 1) time in [0, 1]
        Returns:
            (batch, 2) velocity
        """
        t_emb = self.time_embed(t)  # (batch, time_embed_dim)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


# ═══════════════════════════════════════════════════════════════════════════════
# Rollouts
# ═══════════════════════════════════════════════════════════════════════════════

def sde_rollout(model: nn.Module, x0: torch.Tensor, num_steps: int,
                noise_scale: float) -> torch.Tensor:
    """Euler-Maruyama SDE rollout for training.

    x_{k+1} = x_k + u_theta(x_k, t_k) * dt + sigma * sqrt(dt) * eps_k

    Args:
        model: velocity field
        x0: (batch, 2) initial states
        num_steps: number of Euler-Maruyama steps
        noise_scale: sigma for the diffusion term

    Returns:
        trajectory: (num_steps + 1, batch, 2) full trajectory including x0
    """
    batch = x0.shape[0]
    device = x0.device
    dt = 1.0 / num_steps
    sqrt_dt = math.sqrt(dt)

    trajectory = [x0]
    x = x0
    for k in range(num_steps):
        t_k = k / num_steps
        t = torch.full((batch,), t_k, device=device)
        v = model(x, t)
        noise = noise_scale * sqrt_dt * torch.randn_like(x)
        x = x + v * dt + noise
        trajectory.append(x)
    return torch.stack(trajectory, dim=0)  # (S+1, batch, 2)


def ode_rollout(model: nn.Module, x0: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Euler ODE rollout for evaluation.

    x_{k+1} = x_k + u_theta(x_k, t_k) * dt

    Args:
        model: velocity field
        x0: (batch, 2) initial states
        num_steps: number of Euler steps

    Returns:
        trajectory: (num_steps + 1, batch, 2) full trajectory including x0
    """
    batch = x0.shape[0]
    device = x0.device
    dt = 1.0 / num_steps

    trajectory = [x0]
    x = x0
    for k in range(num_steps):
        t_k = k / num_steps
        t = torch.full((batch,), t_k, device=device)
        with torch.no_grad():
            v = model(x, t)
        x = x + v * dt
        trajectory.append(x)
    return torch.stack(trajectory, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage A: CFM Pretraining
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain_cfm(model: nn.Module, cfg: Config, centers: torch.Tensor) -> nn.Module:
    """Pretrain the velocity model using Conditional Flow Matching.

    CFM objective:
        - Sample (x0, x1) pairs: x0 ~ annulus, x1 ~ 8-Gaussian target
        - Sample t ~ Uniform(0, 1)
        - Interpolate: x_t = (1 - t) * x0 + t * x1
        - Target velocity: v* = x1 - x0
        - Loss: E[||u_theta(x_t, t) - (x1 - x0)||^2]
    """
    device = cfg.device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.pretrain_lr)

    model.train()
    for step in range(cfg.pretrain_steps):
        x0 = sample_annulus(cfg.pretrain_batch_size, cfg.r_min, cfg.r_max, device)
        x1 = sample_target(cfg.pretrain_batch_size, centers, cfg.target_std)

        t = torch.rand(cfg.pretrain_batch_size, device=device)

        # Linear interpolation path
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1

        # Target conditional velocity: v*(x_t, t | x0, x1) = x1 - x0
        v_target = x1 - x0

        # Predicted velocity
        v_pred = model(x_t, t)

        # CFM loss
        loss = ((v_pred - v_target) ** 2).sum(-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 2000 == 0 or step == 0:
            print(f"  [CFM Pretrain] step {step+1}/{cfg.pretrain_steps}, loss={loss.item():.6f}")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Stage B: RL Finetuning
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gcr_loss(model: nn.Module, trajectory: torch.Tensor,
                     num_steps: int, t_max: float) -> torch.Tensor:
    """Compute the Geometric Consistency Regularization loss.

    GCR penalizes the velocity field for deviating from the bridge velocity
    that would transport each intermediate state x_t to the realized terminal
    state x_1 along a straight line:

        v_bridge(x_t, x_1, t) = (x_1 - x_t) / (1 - t)

    L_GCR = E[ sum_k ||u_theta(x_k, t_k) - v_bridge(x_k, x_T, t_k)||^2 * dt ]

    We exclude steps where t >= t_max to avoid the (1 - t) -> 0 singularity.
    """
    dt = 1.0 / num_steps
    x_T = trajectory[-1]  # (batch, 2) — realized terminal state
    device = trajectory.device

    gcr_loss = torch.tensor(0.0, device=device)
    count = 0

    for k in range(num_steps):
        t_k = k / num_steps
        if t_k >= t_max:
            # Skip steps too close to t=1 to avoid numerical blow-up
            # in the bridge velocity denominator (1 - t) -> 0
            break

        x_k = trajectory[k]  # (batch, 2)
        t = torch.full((x_k.shape[0],), t_k, device=device)

        v_pred = model(x_k, t)
        # Bridge velocity: straight-line transport from x_k to realized x_T
        v_bridge = (x_T - x_k) / (1.0 - t_k)

        gcr_loss = gcr_loss + ((v_pred - v_bridge) ** 2).sum(-1).mean() * dt
        count += 1

    return gcr_loss


def finetune_rl(model: nn.Module, cfg: Config, centers: torch.Tensor,
                use_gcr: bool, label: str) -> nn.Module:
    """Finetune the model with RL (and optionally GCR).

    The RL objective maximizes terminal reward through pathwise differentiation
    of the SDE rollout. This is a differentiable surrogate; the code is structured
    so a score-function (REINFORCE-style) estimator could be swapped in by
    replacing the loss computation below.

    Training uses SDE rollouts (Euler-Maruyama with exploration noise).
    """
    device = cfg.device
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.finetune_lr)

    for step in range(cfg.finetune_steps):
        x0 = sample_annulus(cfg.finetune_batch_size, cfg.r_min, cfg.r_max, device)

        # SDE rollout for training — stores full trajectory for GCR
        traj = sde_rollout(model, x0, cfg.num_rollout_steps, cfg.sde_noise_scale)
        x_T = traj[-1]  # (batch, 2)

        # Terminal reward — use assigned mode reward to prevent mode collapse.
        # Each source point is steered toward its angularly nearest rewarded mode.
        reward = assigned_terminal_reward(x_T, x0, centers, cfg.rewarded_modes, cfg.reward_sigma)

        # RL loss: maximize reward via pathwise gradient (differentiable through SDE)
        # Structured so a policy-gradient loss could replace this line.
        loss_rl = -reward.mean()

        # GCR regularization (only for RL+GCR)
        if use_gcr:
            loss_gcr = compute_gcr_loss(model, traj, cfg.num_rollout_steps, cfg.gcr_t_max)
            loss = loss_rl + cfg.lambda_gcr * loss_gcr
        else:
            loss_gcr = torch.tensor(0.0)
            loss = loss_rl

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % 500 == 0 or step == 0:
            print(f"  [{label}] step {step+1}/{cfg.finetune_steps}, "
                  f"reward={reward.mean().item():.4f}, "
                  f"loss_rl={loss_rl.item():.4f}, "
                  f"loss_gcr={loss_gcr.item():.4f}")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_path_energy(trajectory: torch.Tensor) -> float:
    """Compute mean kinetic energy (path action) along ODE trajectories.

    E = (1/N) sum_i sum_k ||x_{k+1}^i - x_k^i||^2 / dt
    """
    # trajectory: (S+1, batch, 2)
    diffs = trajectory[1:] - trajectory[:-1]  # (S, batch, 2)
    num_steps = diffs.shape[0]
    dt = 1.0 / num_steps
    # Kinetic energy per step per trajectory
    step_energy = (diffs ** 2).sum(-1) / dt  # (S, batch)
    # Mean over steps and batch
    return step_energy.mean().item()


def count_crossings(trajectory: torch.Tensor) -> float:
    """Count average pairwise trajectory crossings (tangling metric).

    For each pair of trajectories (i, j), count the number of times
    the line segments of trajectory i cross the line segments of trajectory j.
    Uses a subsampled set for efficiency.

    Returns the average crossing count per pair.
    """
    # trajectory: (S+1, batch, 2) — subsample for efficiency
    traj_np = trajectory.cpu().numpy()
    S, N, _ = traj_np.shape

    # Subsample trajectories and steps for speed
    max_traj = min(N, 50)
    step_stride = max(1, S // 30)
    indices = np.random.choice(N, max_traj, replace=False) if N > max_traj else np.arange(N)
    steps = np.arange(0, S - 1, step_stride)

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4 (2D)."""
        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-12:
            return False
        d3 = p3 - p1
        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
        u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross
        return 0 <= t <= 1 and 0 <= u <= 1

    total_crossings = 0
    num_pairs = 0
    for a_idx in range(len(indices)):
        for b_idx in range(a_idx + 1, len(indices)):
            i, j = indices[a_idx], indices[b_idx]
            for s in steps:
                s_next = min(s + step_stride, S - 1)
                if s == s_next:
                    continue
                p1 = traj_np[s, i]
                p2 = traj_np[s_next, i]
                p3 = traj_np[s, j]
                p4 = traj_np[s_next, j]
                if segments_intersect(p1, p2, p3, p4):
                    total_crossings += 1
            num_pairs += 1

    return total_crossings / max(num_pairs, 1)


def compute_mode_allocation(x_T: torch.Tensor, centers: torch.Tensor) -> dict:
    """Compute endpoint mode allocation statistics.

    Assigns each terminal point to the nearest mode center and reports counts.
    """
    # (batch, K)
    dists = ((x_T.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    assignments = dists.argmin(dim=1)  # (batch,)
    K = centers.shape[0]
    allocation = {}
    for k in range(K):
        allocation[f"mode_{k}"] = int((assignments == k).sum().item())
    return allocation


def compute_neighborhood_preservation(traj: torch.Tensor, k_neighbors: int = 5) -> float:
    """Measure how well local neighborhoods are preserved from start to end.

    For each point, find its k nearest neighbors at t=0 and at t=1, then
    compute the fraction of neighbors that are shared (Jaccard-like).
    """
    x0 = traj[0].cpu().numpy()   # (N, 2)
    x1 = traj[-1].cpu().numpy()  # (N, 2)
    N = x0.shape[0]

    if N <= k_neighbors:
        return 1.0

    # Pairwise distances
    def knn_indices(points, k):
        # (N, N)
        d = np.sum((points[:, None] - points[None, :]) ** 2, axis=-1)
        # Exclude self
        idx = np.argsort(d, axis=1)[:, 1:k+1]
        return idx

    knn0 = knn_indices(x0, k_neighbors)
    knn1 = knn_indices(x1, k_neighbors)

    overlap = 0.0
    for i in range(N):
        s0 = set(knn0[i])
        s1 = set(knn1[i])
        overlap += len(s0 & s1) / len(s0 | s1)
    return overlap / N


def compute_all_metrics(model: nn.Module, cfg: Config, centers: torch.Tensor,
                        x0_fixed: torch.Tensor, label: str) -> dict:
    """Compute all metrics for a model using ODE rollouts."""
    model.eval()
    with torch.no_grad():
        traj = ode_rollout(model, x0_fixed, cfg.num_rollout_steps)

    x_T = traj[-1]
    reward = terminal_reward(x_T, centers, cfg.rewarded_modes, cfg.reward_sigma)

    metrics = {
        "model": label,
        "avg_reward": reward.mean().item(),
        "path_energy": compute_path_energy(traj),
        "crossing_count": count_crossings(traj),
        "neighborhood_preservation": compute_neighborhood_preservation(traj),
    }
    metrics.update({f"alloc_{k}": v for k, v in
                    compute_mode_allocation(x_T, centers).items()})
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(models: dict, cfg: Config, centers: torch.Tensor,
                    x0_plot: torch.Tensor, save_path: str):
    """Generate the 3-column comparison figure.

    Each column shows ODE trajectories colored by initial angle on the annulus,
    terminal scatter, and mode centers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = list(models.keys())

    # Color trajectories by initial angle
    angles = torch.atan2(x0_plot[:, 1], x0_plot[:, 0]).cpu().numpy()
    angles = (angles + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    cmap = cm.hsv

    all_centers = centers.cpu().numpy()
    rewarded_mask = np.zeros(len(all_centers), dtype=bool)
    for idx in cfg.rewarded_modes:
        rewarded_mask[idx] = True

    for col, (label, model) in enumerate(models.items()):
        ax = axes[col]
        model.eval()
        with torch.no_grad():
            traj = ode_rollout(model, x0_plot, cfg.num_rollout_steps)
        traj_np = traj.cpu().numpy()  # (S+1, N, 2)
        x_T_np = traj_np[-1]

        # Plot trajectories
        for i in range(traj_np.shape[1]):
            color = cmap(angles[i])
            ax.plot(traj_np[:, i, 0], traj_np[:, i, 1],
                    color=color, alpha=0.3, linewidth=0.5)

        # Terminal scatter
        for i in range(x_T_np.shape[0]):
            ax.scatter(x_T_np[i, 0], x_T_np[i, 1],
                       color=cmap(angles[i]), s=4, alpha=0.6, zorder=3)

        # Mode centers
        ax.scatter(all_centers[~rewarded_mask, 0], all_centers[~rewarded_mask, 1],
                   marker="x", s=80, c="gray", linewidths=2, zorder=5, label="Unrewarded modes")
        ax.scatter(all_centers[rewarded_mask, 0], all_centers[rewarded_mask, 1],
                   marker="*", s=200, c="red", edgecolors="darkred", linewidths=0.5,
                   zorder=5, label="Rewarded modes")

        # Source annulus ring (visual reference)
        theta_ring = np.linspace(0, 2 * np.pi, 200)
        for r in [cfg.r_min, cfg.r_max]:
            ax.plot(r * np.cos(theta_ring), r * np.sin(theta_ring),
                    "k--", alpha=0.15, linewidth=0.5)

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        if col == 0:
            ax.legend(loc="lower left", fontsize=7)

    fig.suptitle("CFM Pretrained → RL Finetuning: Base vs RL-only vs RL+GCR",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = config_from_args()

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    print(f"Device: {device}")
    print(f"Config: {vars(cfg)}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Distributions ──────────────────────────────────────────────────────
    centers = get_mode_centers(cfg.num_modes, cfg.target_radius, device)
    print(f"Mode centers (K={cfg.num_modes}):")
    for i, c in enumerate(centers):
        marker = " ← REWARDED" if i in cfg.rewarded_modes else ""
        print(f"  mode {i}: ({c[0].item():.3f}, {c[1].item():.3f}){marker}")
    print()

    # ── Fixed evaluation samples ───────────────────────────────────────────
    # Use the same initial points for all three models for fair comparison
    torch.manual_seed(cfg.seed + 999)
    x0_eval = sample_annulus(cfg.num_eval_samples, cfg.r_min, cfg.r_max, device)
    x0_plot = sample_annulus(cfg.num_plot_trajectories, cfg.r_min, cfg.r_max, device)
    torch.manual_seed(cfg.seed)  # reset

    # ── Stage A: Pretrain with CFM ─────────────────────────────────────────
    print("=" * 60)
    print("STAGE A: Conditional Flow Matching Pretraining")
    print("=" * 60)
    base_model = VelocityModel(
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
    ).to(device)
    base_model = pretrain_cfm(base_model, cfg, centers)
    print()

    # Save pretrained checkpoint
    pretrain_path = os.path.join(cfg.output_dir, "pretrained.pt")
    torch.save(base_model.state_dict(), pretrain_path)
    print(f"  Pretrained model saved to {pretrain_path}\n")

    # ── Stage B: RL Finetuning ─────────────────────────────────────────────
    # Clone the pretrained model for each finetuning variant
    rl_only_model = copy.deepcopy(base_model)
    rl_gcr_model = copy.deepcopy(base_model)

    print("=" * 60)
    print("STAGE B: RL Finetuning (RL-only, no GCR)")
    print("=" * 60)
    torch.manual_seed(cfg.seed + 1)
    rl_only_model = finetune_rl(rl_only_model, cfg, centers,
                                use_gcr=False, label="RL-only")
    print()

    print("=" * 60)
    print("STAGE B: RL Finetuning (RL+GCR)")
    print("=" * 60)
    torch.manual_seed(cfg.seed + 1)  # same seed for fair comparison
    rl_gcr_model = finetune_rl(rl_gcr_model, cfg, centers,
                               use_gcr=True, label="RL+GCR")
    print()

    # Save finetuned checkpoints
    torch.save(rl_only_model.state_dict(),
               os.path.join(cfg.output_dir, "rl_only.pt"))
    torch.save(rl_gcr_model.state_dict(),
               os.path.join(cfg.output_dir, "rl_gcr.pt"))
    print("  Finetuned models saved.\n")

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("EVALUATION (ODE rollouts)")
    print("=" * 60)

    models = {
        "Base (CFM pretrained)": base_model,
        "RL-only": rl_only_model,
        "RL+GCR": rl_gcr_model,
    }

    all_metrics = []
    for label, model in models.items():
        m = compute_all_metrics(model, cfg, centers, x0_eval, label)
        all_metrics.append(m)
        print(f"\n  {label}:")
        print(f"    avg_reward:       {m['avg_reward']:.4f}")
        print(f"    path_energy:      {m['path_energy']:.4f}")
        print(f"    crossing_count:   {m['crossing_count']:.2f}")
        print(f"    nbhd_preserv:     {m['neighborhood_preservation']:.4f}")
        alloc_str = ", ".join(f"m{i}={m.get(f'alloc_mode_{i}', 0)}"
                              for i in range(cfg.num_modes))
        print(f"    mode_allocation:  {alloc_str}")
    print()

    # Save metrics
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # ── Visualization ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    fig_path = os.path.join(cfg.output_dir, "comparison.png")
    plot_comparison(models, cfg, centers, x0_plot, fig_path)

    # Save config for reproducibility
    config_path = os.path.join(cfg.output_dir, "config.json")
    cfg_dict = vars(cfg)
    cfg_dict["device"] = str(cfg_dict.get("device", device))
    with open(config_path, "w") as f:
        json.dump({k: v for k, v in cfg_dict.items()
                   if not callable(v)}, f, indent=2, default=str)
    print(f"  Config saved to {config_path}")

    print("\n" + "=" * 60)
    print("DONE. All outputs in:", cfg.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
