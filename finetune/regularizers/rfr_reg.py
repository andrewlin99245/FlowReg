from __future__ import annotations

import torch

from .base import BaseRegularizer, RegularizerInputs


class RFRRegularizer(BaseRegularizer):
    name = "rfr"

    def compute(self, inputs: RegularizerInputs) -> torch.Tensor:
        time_view = torch.clamp(1.0 - inputs.times, min=1e-4).view(-1, 1, 1, 1)
        target = (inputs.expanded_terminal_states - inputs.x_t) / time_view
        return (inputs.predicted_velocity - target).square().flatten(1).sum(dim=1).mean()
