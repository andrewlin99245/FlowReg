from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parents[1].name == "_project_shared_stage":
    FLOWREG_ROOT = THIS_FILE.parents[3]
else:
    FLOWREG_ROOT = THIS_FILE.parents[1]
PRETRAIN_ROOT = FLOWREG_ROOT / "pretrain"
FINETUNE_ROOT = FLOWREG_ROOT / "finetune"
if str(PRETRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_ROOT))
if str(FINETUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(FINETUNE_ROOT))

from dataset import CLASS_RECORDS  # type: ignore  # noqa: E402
from rewards.classifier_reward import ImageNetClassifierReward  # type: ignore  # noqa: E402
from rewards.musiq_reward import MUSIQReward  # type: ignore  # noqa: E402
from rewards.reward_factory import build_reward_function  # type: ignore  # noqa: E402
from utils.config import load_config  # type: ignore  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load FlowReg reward models and validate class mapping.")
    parser.add_argument("--config", type=str, default=str(FINETUNE_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PRETRAIN_ROOT / "outputs" / "reward_bootstrap" / "reward_bootstrap_summary.json"),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    summary: dict[str, object] = {
        "config": str(Path(args.config).resolve()),
        "device": args.device,
        "class_mapping": [
            {
                "local_index": int(record.local_index),
                "requested_name": record.requested_name,
                "imagenet_name": record.imagenet_name,
                "global_index": int(record.global_index),
            }
            for record in CLASS_RECORDS
        ],
    }

    classifier = ImageNetClassifierReward(config["reward"]["classifier"], device=device)
    summary["classifier"] = {
        "loaded": True,
        "backend": config["reward"]["classifier"]["backend"],
        "label_space": config["reward"]["classifier"]["label_space"],
        "num_outputs": int(classifier.num_outputs),
    }

    musiq = MUSIQReward(config["reward"]["musiq"], device=device)
    summary["musiq"] = {
        "loaded": True,
        "metric_name": config["reward"]["musiq"]["metric_name"],
    }

    reward_cfg = json.loads(json.dumps(config["reward"]))
    reward_cfg["setting"] = "classifier_plus_musiq"
    reward_fn = build_reward_function(reward_cfg, device=device)
    images = torch.randn(2, 3, 64, 64, device=device)
    labels = torch.tensor([0, 1], device=device, dtype=torch.long)
    outputs = reward_fn(images, labels)
    summary["forward_pass"] = {
        "total_shape": list(outputs.total.shape),
        "classifier_shape": list(outputs.classifier.shape),
        "musiq_shape": None if outputs.musiq is None else list(outputs.musiq.shape),
        "total_mean": float(outputs.total.mean().cpu()),
        "classifier_mean": float(outputs.classifier.mean().cpu()),
        "musiq_mean": None if outputs.musiq is None else float(outputs.musiq.mean().cpu()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Saved reward bootstrap summary to {output_path}")


if __name__ == "__main__":
    main()
