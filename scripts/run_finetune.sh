#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${FINETUNE_ROOT}"

if [[ $# -lt 1 ]]; then
  set -- configs/experiments/no_reg_classifier.yaml
fi

CONFIG_PATH="$1"
shift

flowreg_python train_finetune.py --config "${CONFIG_PATH}" "$@"
