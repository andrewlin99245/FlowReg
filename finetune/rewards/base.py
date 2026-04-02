from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardOutputs:
    total: torch.Tensor
    classifier: torch.Tensor
    musiq: torch.Tensor | None = None


class BaseRewardFunction:
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> RewardOutputs:
        raise NotImplementedError
