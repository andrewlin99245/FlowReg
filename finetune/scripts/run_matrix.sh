#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/experiments/no_reg_classifier.yaml
  configs/experiments/no_reg_classifier_plus_musiq.yaml
  configs/experiments/w2_classifier.yaml
  configs/experiments/w2_classifier_plus_musiq.yaml
  configs/experiments/rfr_classifier.yaml
  configs/experiments/rfr_classifier_plus_musiq.yaml
  configs/experiments/batchot_classifier.yaml
  configs/experiments/batchot_classifier_plus_musiq.yaml
)

for config in "${configs[@]}"; do
  python train_finetune.py --config "${config}"
done
