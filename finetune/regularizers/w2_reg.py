from __future__ import annotations

import torch

from .base import BaseRegularizer, RegularizerInputs


class W2Regularizer(BaseRegularizer):
    name = "w2"

    def compute(self, inputs: RegularizerInputs) -> torch.Tensor:
        return inputs.predicted_velocity.square().flatten(1).sum(dim=1).mean()
