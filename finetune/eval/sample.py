from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from models import deterministic_euler_sample, load_flow_checkpoint
from utils import load_config, select_device, slugify
from utils.misc import configure_cuda_runtime


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate class-conditional samples from a fine-tuned checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    return parser


@torch.no_grad()
def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    device = select_device(config.get("device", "auto"))
    configure_cuda_runtime(device)
    model, metadata = load_flow_checkpoint(args.checkpoint, device=device, freeze=True)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["root"]) / config["experiment_name"] / "eval_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config["eval"]
    class_ids = eval_cfg.get("class_ids", []) or list(range(len(metadata["requested_class_names"])))
    samples_per_class = int(eval_cfg["samples_per_class"])
    nrow = int(samples_per_class**0.5)
    if nrow * nrow != samples_per_class:
        nrow = min(4, samples_per_class)

    for class_id in class_ids:
        labels = torch.full((samples_per_class,), int(class_id), device=device, dtype=torch.long)
        samples = deterministic_euler_sample(
            model=model,
            labels=labels,
            num_steps=int(eval_cfg["sample_steps"]),
            device=device,
            image_shape=tuple(config["model"]["image_shape"]),
        )
        grid = make_grid(samples.clamp(-1.0, 1.0).add(1.0).div(2.0), nrow=nrow)
        class_name = metadata["requested_class_names"][int(class_id)]
        save_image(grid, output_dir / f"{int(class_id):02d}_{slugify(class_name)}.png")


if __name__ == "__main__":
    main()
