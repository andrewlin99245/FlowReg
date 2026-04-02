from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RegularizerInputs:
    predicted_velocity: torch.Tensor
    x_t: torch.Tensor
    times: torch.Tensor
    labels: torch.Tensor
    terminal_states: torch.Tensor


class BaseRegularizer:
    name = "base"

    def __init__(self, weight: float):
        self.weight = float(weight)

    def compute(self, inputs: RegularizerInputs) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    def update_reference_batch(self, terminal_states: torch.Tensor) -> None:
        del terminal_states
