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
    repeats_per_terminal: int = 1
    anchor_states: torch.Tensor | None = None

    def expand_condition_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.shape[0] == self.predicted_velocity.shape[0]:
            return tensor
        if self.repeats_per_terminal <= 1:
            raise ValueError("Condition tensor shape does not match the rollout batch.")
        if self.predicted_velocity.shape[0] != tensor.shape[0] * self.repeats_per_terminal:
            raise ValueError("Condition tensor cannot be expanded to match the rollout batch.")
        return tensor.repeat_interleave(self.repeats_per_terminal, dim=0)

    @property
    def expanded_terminal_states(self) -> torch.Tensor:
        expanded = self.expand_condition_tensor(self.terminal_states)
        if expanded is None:
            raise ValueError("terminal_states must be present for this regularizer.")
        return expanded

    @property
    def expanded_anchor_states(self) -> torch.Tensor | None:
        return self.expand_condition_tensor(self.anchor_states)


class BaseRegularizer:
    name = "base"

    def __init__(self, weight: float):
        self.weight = float(weight)

    def compute(self, inputs: RegularizerInputs) -> torch.Tensor:
        raise NotImplementedError

    def prepare_rollout(self, terminal_states: torch.Tensor) -> None:
        del terminal_states

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    def update_reference_batch(self, terminal_states: torch.Tensor) -> None:
        del terminal_states

    def select_anchor_states(
        self,
        trajectory_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        del trajectory_indices, device
        return None
