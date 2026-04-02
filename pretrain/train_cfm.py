from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import PreparedImageNet64Subset, TARGET_CLASS_NAMES, is_prepared_dataset, load_prepared_metadata
from model import ClassConditionedUNet
from prepare_imagenet64_subset import DEFAULT_DATASET_NAME, prepare_dataset
from sampling import autocast_context, resolve_amp_settings, save_class_conditional_grids


DEFAULT_CHECKPOINT_STEPS = (50000, 100000, 150000, 200000, 250000, 300000)


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unable to parse boolean value: {value}")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_loader(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


class WarmupCosineScheduler:
    def __init__(self, optimizer: optim.Optimizer, base_lr: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        self._apply(self.get_lr(0))

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * cosine

    def _apply(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.step_count += 1
        self._apply(self.get_lr(self.step_count))

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step_count = int(state_dict["step_count"])
        self._apply(self.get_lr(self.step_count))

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


@dataclass
class TrainConfig:
    data_root: str = "data/imagenet64_subset50"
    cache_dir: str = "data/hf_cache"
    dataset_name: str = DEFAULT_DATASET_NAME
    output_dir: str = "outputs/imagenet64_subset50_cfm"
    auto_prepare: bool = True
    force_prepare: bool = False
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 2e-4
    warmup_steps: int = 5000
    max_steps: int = 300000
    grad_clip: float = 1.0
    mixed_precision: bool = True
    sample_every: int = 10000
    sample_steps: int = 100
    samples_per_class: int = 16
    checkpoint_steps: tuple[int, ...] = field(default_factory=lambda: DEFAULT_CHECKPOINT_STEPS)
    log_every: int = 100
    resume: str = ""


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Class-conditioned CFM pretraining on ImageNet64-50.")
    parser.add_argument("--data-root", type=str, default="data/imagenet64_subset50")
    parser.add_argument("--cache-dir", type=str, default="data/hf_cache")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=str, default="outputs/imagenet64_subset50_cfm")
    parser.add_argument("--auto-prepare", type=parse_bool, default=True)
    parser.add_argument("--force-prepare", type=parse_bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=300000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=parse_bool, default=True)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--checkpoint-steps", type=parse_int_list, default=DEFAULT_CHECKPOINT_STEPS)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=str, default="")
    return parser


def config_from_args() -> TrainConfig:
    args = build_argparser().parse_args()
    return TrainConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        auto_prepare=args.auto_prepare,
        force_prepare=args.force_prepare,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        sample_every=args.sample_every,
        sample_steps=args.sample_steps,
        samples_per_class=args.samples_per_class,
        checkpoint_steps=args.checkpoint_steps,
        log_every=args.log_every,
        resume=args.resume,
    )


def ensure_prepared_dataset(config: TrainConfig) -> None:
    if is_prepared_dataset(config.data_root) and not config.force_prepare:
        return
    if not config.auto_prepare and not is_prepared_dataset(config.data_root):
        raise FileNotFoundError(
            f"Prepared dataset not found at {config.data_root}. "
            "Either enable --auto-prepare true or run prepare_imagenet64_subset.py first."
        )
    prepare_dataset(
        cache_dir=config.cache_dir,
        prepared_root=config.data_root,
        dataset_name=config.dataset_name,
        force=config.force_prepare,
    )


def make_dataloader(config: TrainConfig, split: str) -> DataLoader:
    dataset = PreparedImageNet64Subset(config.data_root, split=split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=config.num_workers > 0,
    )


def compute_cfm_loss(model: nn.Module, x1: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=torch.float32)
    t_view = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0
    predicted_velocity = model(x_t, t, labels)
    return F.mse_loss(predicted_velocity, target_velocity)


def collect_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_json(path: str | Path, payload: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_metric(path: str | Path, payload: dict[str, object]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    config: TrainConfig,
    metadata: dict[str, object],
) -> None:
    checkpoint = {
        "step": step,
        "config": asdict(config),
        "metadata": metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": collect_rng_state(),
    }
    torch.save(checkpoint, path)


def maybe_resume(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["step"])


def main() -> None:
    config = config_from_args()
    set_seed(config.seed)
    ensure_prepared_dataset(config)

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    prepared_metadata = load_prepared_metadata(config.data_root)
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "dataset_metadata.json", prepared_metadata)

    device = select_device()
    configure_cuda_runtime(device)
    amp_settings = resolve_amp_settings(device, enabled=config.mixed_precision)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_settings.enabled)

    train_loader = make_dataloader(config, split="train")
    train_iterator = infinite_loader(train_loader)

    model = ClassConditionedUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        base_lr=config.lr,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )

    latest_checkpoint = checkpoint_dir / "latest.pt"
    global_step = 0
    if config.resume:
        global_step = maybe_resume(config.resume, model, optimizer, scheduler, scaler)
    elif latest_checkpoint.is_file():
        global_step = maybe_resume(latest_checkpoint, model, optimizer, scheduler, scaler)

    checkpoint_steps = set(config.checkpoint_steps)
    metrics_path = output_dir / "metrics.jsonl"
    step_timer = time.time()
    metadata = {
        "requested_class_names": TARGET_CLASS_NAMES,
        "prepared_data_root": config.data_root,
    }

    while global_step < config.max_steps:
        images, labels, _, _ = next(train_iterator)
        images = images.to(device, non_blocking=torch.cuda.is_available())
        labels = labels.to(device, non_blocking=torch.cuda.is_available())

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_settings):
            loss = compute_cfm_loss(model, images, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        if global_step % config.log_every == 0 or global_step == 1:
            elapsed = time.time() - step_timer
            metric = {
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "lr": scheduler.lr,
                "steps_per_second": float(config.log_every / elapsed) if global_step > 1 else 0.0,
            }
            append_metric(metrics_path, metric)
            print(
                f"step={global_step} loss={metric['loss']:.6f} lr={metric['lr']:.6e} "
                f"steps_per_second={metric['steps_per_second']:.2f}"
            )
            step_timer = time.time()

        if global_step % config.sample_every == 0 or global_step in checkpoint_steps:
            save_class_conditional_grids(
                model=model,
                output_dir=sample_dir,
                class_names=prepared_metadata["requested_class_names"],
                step=global_step,
                samples_per_class=config.samples_per_class,
                num_steps=config.sample_steps,
                device=device,
                amp_settings=amp_settings,
            )

        if global_step in checkpoint_steps:
            save_checkpoint(
                path=checkpoint_dir / f"step_{global_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=global_step,
                config=config,
                metadata=metadata,
            )
        if global_step % config.log_every == 0 or global_step in checkpoint_steps:
            save_checkpoint(
                path=latest_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=global_step,
                config=config,
                metadata=metadata,
            )

    print(f"Training finished at step {global_step}.")


if __name__ == "__main__":
    main()
