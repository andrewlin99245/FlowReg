from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from models.checkpoint import LOCAL_TO_GLOBAL


@dataclass
class ClassifierConfig:
    backend: str
    checkpoint_path: str
    label_space: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class ImageNetClassifierReward:
    def __init__(self, config: dict, device: torch.device):
        self.config = ClassifierConfig(
            backend=str(config.get("backend", "torchvision")),
            checkpoint_path=str(config.get("checkpoint_path", "")),
            label_space=str(config.get("label_space", "imagenet1k")),
            input_size=int(config.get("input_size", 224)),
            mean=tuple(config.get("mean", [0.485, 0.456, 0.406])),
            std=tuple(config.get("std", [0.229, 0.224, 0.225])),
        )
        if self.config.backend != "torchvision":
            raise ValueError(f"Unsupported classifier backend: {self.config.backend}")
        weights = None if self.config.checkpoint_path else ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        if self.config.checkpoint_path:
            state_dict = torch.load(self.config.checkpoint_path, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
        self.model = _freeze_module(model.to(device))
        self.device = device
        self.mean = torch.tensor(self.config.mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(self.config.std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.local_to_global = {int(key): int(value) for key, value in LOCAL_TO_GLOBAL.items()}
        self.num_outputs = self._infer_num_outputs(self.model)
        self._validate_label_space()

    @staticmethod
    def _infer_num_outputs(model: torch.nn.Module) -> int:
        if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
            return int(model.fc.out_features)
        if hasattr(model, "classifier"):
            classifier = model.classifier
            if hasattr(classifier, "out_features"):
                return int(classifier.out_features)
            if isinstance(classifier, torch.nn.Sequential) and len(classifier) > 0 and hasattr(classifier[-1], "out_features"):
                return int(classifier[-1].out_features)
        if hasattr(model, "head") and hasattr(model.head, "out_features"):
            return int(model.head.out_features)
        raise ValueError("Unable to infer classifier output dimension for reward model.")

    def _validate_label_space(self) -> None:
        if self.config.label_space == "imagenet1k":
            required = max(self.local_to_global.values()) + 1
            if self.num_outputs < required:
                raise ValueError(
                    f"Classifier only has {self.num_outputs} outputs, but imagenet1k mapping requires at least {required}."
                )
            return
        if self.config.label_space == "local50":
            if self.num_outputs != len(self.local_to_global):
                raise ValueError(
                    "label_space=local50 requires a 50-way classifier checkpoint. "
                    "For the default torchvision ImageNet-1k classifier, keep label_space=imagenet1k."
                )
            return
        raise ValueError(f"Unsupported classifier label_space: {self.config.label_space}")

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.clamp(-1.0, 1.0).add(1.0).div(2.0)
        images = F.interpolate(images, size=(self.config.input_size, self.config.input_size), mode="bilinear", align_corners=False)
        return (images - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized = self._preprocess(images.to(self.device, dtype=torch.float32))
        logits = self.model(normalized)
        probabilities = logits.softmax(dim=1)
        if self.config.label_space == "imagenet1k":
            target_indices = torch.tensor(
                [self.local_to_global[int(label)] for label in labels.detach().cpu().tolist()],
                device=self.device,
                dtype=torch.long,
            )
        elif self.config.label_space == "local50":
            target_indices = labels.to(self.device, dtype=torch.long)
        return probabilities.gather(1, target_indices.view(-1, 1)).squeeze(1)
