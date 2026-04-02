#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PRETRAIN_ROOT}"
flowreg_python prepare_imagenet64_subset.py \
  --cache-dir data/hf_cache \
  --prepared-root data/imagenet64_subset50 \
  "$@"
