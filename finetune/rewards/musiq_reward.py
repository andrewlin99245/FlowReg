from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MUSIQConfig:
    metric_name: str
    checkpoint_path: str
    raw_min: float
    raw_max: float


def _freeze_metric(metric: torch.nn.Module) -> torch.nn.Module:
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


class MUSIQReward:
    def __init__(self, config: dict, device: torch.device):
        try:
            import pyiqa
        except ImportError as exc:
            raise ImportError("pyiqa is required for MUSIQ reward support.") from exc
        self.config = MUSIQConfig(
            metric_name=str(config.get("metric_name", "musiq")),
            checkpoint_path=str(config.get("checkpoint_path", "")),
            raw_min=float(config.get("raw_min", 0.0)),
            raw_max=float(config.get("raw_max", 100.0)),
        )
        metric = pyiqa.create_metric(self.config.metric_name, device=device)
        if self.config.checkpoint_path:
            state_dict = torch.load(self.config.checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if hasattr(metric, "load_state_dict"):
                metric.load_state_dict(state_dict)
            elif hasattr(metric, "net") and hasattr(metric.net, "load_state_dict"):
                metric.net.load_state_dict(state_dict)
            else:
                raise ValueError("Unable to load MUSIQ checkpoint into pyiqa metric.")
        self.metric = _freeze_metric(metric.to(device))
        self.device = device

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.clamp(-1.0, 1.0).add(1.0).div(2.0).to(self.device, dtype=torch.float32)
        raw_scores = self.metric(images).view(-1)
        normalized = (raw_scores - self.config.raw_min) / max(self.config.raw_max - self.config.raw_min, 1e-6)
        return raw_scores, normalized.clamp_(0.0, 1.0)
