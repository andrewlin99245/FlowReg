from __future__ import annotations

import math

import torch


@torch.no_grad()
def sinkhorn_uniform_transport(cost: torch.Tensor, epsilon: float, num_iters: int) -> torch.Tensor:
    batch_size = int(cost.shape[0])
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("Sinkhorn currently expects a square cost matrix.")
    log_mass = -math.log(float(batch_size))
    log_a = torch.full((batch_size,), log_mass, device=cost.device, dtype=cost.dtype)
    log_b = log_a.clone()
    kernel = -cost / max(float(epsilon), 1e-8)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(kernel + v.view(1, -1), dim=1)
        v = log_b - torch.logsumexp(kernel + u.view(-1, 1), dim=0)
    return torch.exp(kernel + u.view(-1, 1) + v.view(1, -1))
