from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


@dataclass(frozen=True)
class AMPSettings:
    enabled: bool
    device_type: str
    dtype: torch.dtype | None


def resolve_amp_settings(device: torch.device, enabled: bool = True) -> AMPSettings:
    if not enabled:
        return AMPSettings(enabled=False, device_type=device.type, dtype=None)
    if device.type in {"cuda", "mps"}:
        return AMPSettings(enabled=True, device_type=device.type, dtype=torch.float16)
    if device.type == "cpu":
        return AMPSettings(enabled=True, device_type=device.type, dtype=torch.bfloat16)
    return AMPSettings(enabled=False, device_type=device.type, dtype=None)


def autocast_context(amp_settings: AMPSettings):
    if not amp_settings.enabled or amp_settings.dtype is None:
        return nullcontext()
    return torch.amp.autocast(
        device_type=amp_settings.device_type,
        dtype=amp_settings.dtype,
        enabled=amp_settings.enabled,
    )


def slugify_class_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    labels: torch.Tensor,
    num_steps: int = 100,
    image_shape: tuple[int, int, int] = (3, 64, 64),
    device: torch.device | None = None,
    amp_settings: AMPSettings | None = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    if amp_settings is None:
        amp_settings = resolve_amp_settings(device, enabled=True)

    batch_size = int(labels.shape[0])
    x = torch.randn(batch_size, *image_shape, device=device)
    dt = 1.0 / float(num_steps)

    for step in range(num_steps):
        t = torch.full((batch_size,), step * dt, device=device, dtype=torch.float32)
        with autocast_context(amp_settings):
            velocity = model(x, t, labels)
        x = x + dt * velocity.float()
    return x


@torch.no_grad()
def save_class_conditional_grids(
    model: torch.nn.Module,
    output_dir: str | Path,
    class_names: list[str],
    step: int,
    samples_per_class: int = 16,
    num_steps: int = 100,
    device: torch.device | None = None,
    amp_settings: AMPSettings | None = None,
) -> None:
    if device is None:
        device = next(model.parameters()).device
    if amp_settings is None:
        amp_settings = resolve_amp_settings(device, enabled=True)

    output_path = Path(output_dir) / f"step_{step:06d}"
    output_path.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()

    nrow = int(math.sqrt(samples_per_class))
    if nrow * nrow != samples_per_class:
        nrow = min(4, samples_per_class)

    for local_label, class_name in enumerate(class_names):
        labels = torch.full((samples_per_class,), local_label, device=device, dtype=torch.long)
        samples = euler_sample(
            model=model,
            labels=labels,
            num_steps=num_steps,
            device=device,
            amp_settings=amp_settings,
        )
        images = samples.clamp(-1.0, 1.0).add(1.0).div(2.0)
        grid = make_grid(images, nrow=nrow)
        save_image(grid, output_path / f"{local_label:02d}_{slugify_class_name(class_name)}.png")

    if was_training:
        model.train()
