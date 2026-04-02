from __future__ import annotations

import math
import os
import random
from typing import Any

import numpy as np
import torch


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.step_count = 0
        self._apply(self.get_lr(0))

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return self.base_lr
        progress = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.step_count += 1
        self._apply(self.get_lr(self.step_count))

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step_count = int(state_dict["step_count"])
        self._apply(self.get_lr(self.step_count))

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


def resolve_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))
