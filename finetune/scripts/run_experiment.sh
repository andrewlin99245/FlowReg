#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <config-path>"
  exit 1
fi

python train_finetune.py --config "$1"
