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
        self.current_anchor_states: torch.Tensor | None = None

    def _compute_anchors(self, terminal_states: torch.Tensor) -> torch.Tensor | None:
        if self.previous_terminal_states is None:
            return None
        current = terminal_states.detach().flatten(1).float()
        previous_states = self.previous_terminal_states.detach().to(device=current.device, dtype=torch.float32)
        previous = previous_states.flatten(1)
        current_sq = current.square().sum(dim=1, keepdim=True)
        previous_sq = previous.square().sum(dim=1).view(1, -1)
        cost = (current_sq + previous_sq - 2.0 * (current @ previous.t())).clamp_min_(0.0)
        plan = sinkhorn_uniform_transport(cost=cost, epsilon=self.epsilon, num_iters=self.num_iters)
        plan = plan / torch.clamp(plan.sum(dim=1, keepdim=True), min=1e-8)
        anchors = plan @ previous
        return anchors.view_as(previous_states)

    @torch.no_grad()
    def prepare_rollout(self, terminal_states: torch.Tensor) -> None:
        anchors = self._compute_anchors(terminal_states)
        self.current_anchor_states = None if anchors is None else anchors.detach()

    def select_anchor_states(
        self,
        trajectory_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.current_anchor_states is None:
            return None
        selected = self.current_anchor_states.index_select(0, trajectory_indices.to(self.current_anchor_states.device))
        return selected.to(device=device)

    def compute(self, inputs: RegularizerInputs) -> torch.Tensor:
        anchors = inputs.expanded_anchor_states
        if anchors is None:
            anchors = self._compute_anchors(inputs.terminal_states)
            anchors = inputs.expand_condition_tensor(anchors)
        if anchors is None:
            return torch.zeros((), device=inputs.predicted_velocity.device, dtype=inputs.predicted_velocity.dtype)
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
        self.current_anchor_states = None
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.num_iters = int(state_dict.get("num_iters", self.num_iters))

    def update_reference_batch(self, terminal_states: torch.Tensor) -> None:
        self.current_anchor_states = None
        self.previous_terminal_states = terminal_states.detach().cpu().clone()
