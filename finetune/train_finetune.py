from __future__ import annotations

import argparse

from trainers import FlowGRPOTrainer
from utils import load_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flow-GRPO fine-tuning for class-conditioned flow models.")
    parser.add_argument("--config", type=str, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    trainer = FlowGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
