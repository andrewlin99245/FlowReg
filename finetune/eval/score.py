from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from models import deterministic_euler_sample, load_flow_checkpoint
from rewards import build_reward_function
from utils import load_config, save_json, select_device
from utils.misc import configure_cuda_runtime


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a fine-tuned checkpoint with the configured reward models.")
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
    reward_fn = build_reward_function(config["reward"], device=device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["root"]) / config["experiment_name"] / "eval_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config["eval"]
    class_ids = eval_cfg.get("class_ids", []) or list(range(len(metadata["requested_class_names"])))
    samples_per_class = int(eval_cfg["samples_per_class"])
    records: list[dict[str, float | int | str | None]] = []

    for class_id in class_ids:
        labels = torch.full((samples_per_class,), int(class_id), device=device, dtype=torch.long)
        samples = deterministic_euler_sample(
            model=model,
            labels=labels,
            num_steps=int(eval_cfg["sample_steps"]),
            device=device,
            image_shape=tuple(config["model"]["image_shape"]),
        )
        rewards = reward_fn(samples, labels)
        records.append(
            {
                "class_id": int(class_id),
                "class_name": metadata["requested_class_names"][int(class_id)],
                "reward_total_mean": float(rewards.total.mean().cpu()),
                "reward_classifier_mean": float(rewards.classifier.mean().cpu()),
                "reward_musiq_mean": None if rewards.musiq is None else float(rewards.musiq.mean().cpu()),
            }
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_classes": len(records),
        "reward_total_mean": float(sum(float(record["reward_total_mean"]) for record in records) / max(len(records), 1)),
        "reward_classifier_mean": float(sum(float(record["reward_classifier_mean"]) for record in records) / max(len(records), 1)),
        "reward_musiq_mean": None
        if all(record["reward_musiq_mean"] is None for record in records)
        else float(
            sum(float(record["reward_musiq_mean"] or 0.0) for record in records) / max(len(records), 1)
        ),
        "per_class": records,
    }
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


if __name__ == "__main__":
    main()
