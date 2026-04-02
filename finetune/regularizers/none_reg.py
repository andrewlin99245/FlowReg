from __future__ import annotations

import torch

from .base import BaseRegularizer, RegularizerInputs


class NoRegularizer(BaseRegularizer):
    name = "no_reg"

    def __init__(self):
        super().__init__(weight=0.0)

    def compute(self, inputs: RegularizerInputs) -> torch.Tensor:
        return torch.zeros((), device=inputs.predicted_velocity.device, dtype=inputs.predicted_velocity.dtype)
