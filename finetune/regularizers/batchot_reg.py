from __future__ import annotations

from typing import Any

import torch

from .base import BaseRegularizer, RegularizerInputs
from .sinkhorn import sinkhorn_uniform_transport


class BatchOTRegularizer(BaseRegularizer):
    name = "batchot"

    def __init__(self, weight: float, epsilon: float, num_iters: int):
        super().__init__(weight=weight)
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.previous_terminal_states: torch.Tensor | None = None

    def _compute_anchors(self, terminal_states: torch.Tensor) -> torch.Tensor | None:
        if self.previous_terminal_states is None:
            return None
        current = terminal_states.detach().flatten(1)
        previous = self.previous_terminal_states.detach().flatten(1).to(current.device)
        cost = torch.cdist(current, previous, p=2).square()
        plan = sinkhorn_uniform_transport(cost=cost, epsilon=self.epsilon, num_iters=self.num_iters)
        plan = plan / torch.clamp(plan.sum(dim=1, keepdim=True), min=1e-8)
        return plan @ self.previous_terminal_states.detach().to(current.device).flatten(1)

    def compute(self, inputs: RegularizerInputs) -> torch.Tensor:
        anchors = self._compute_anchors(inputs.terminal_states)
        if anchors is None:
            return torch.zeros((), device=inputs.predicted_velocity.device, dtype=inputs.predicted_velocity.dtype)
        anchors = anchors.view_as(inputs.terminal_states)
        time_view = torch.clamp(1.0 - inputs.times, min=1e-4).view(-1, 1, 1, 1)
        target = (anchors - inputs.x_t) / time_view
        return (inputs.predicted_velocity - target).square().flatten(1).sum(dim=1).mean()

    def state_dict(self) -> dict[str, Any]:
        return {
            "previous_terminal_states": None
            if self.previous_terminal_states is None
            else self.previous_terminal_states.detach().cpu(),
            "epsilon": self.epsilon,
            "num_iters": self.num_iters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        terminal_states = state_dict.get("previous_terminal_states")
        self.previous_terminal_states = None if terminal_states is None else terminal_states.detach().clone()
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.num_iters = int(state_dict.get("num_iters", self.num_iters))

    def update_reference_batch(self, terminal_states: torch.Tensor) -> None:
        self.previous_terminal_states = terminal_states.detach().cpu().clone()
