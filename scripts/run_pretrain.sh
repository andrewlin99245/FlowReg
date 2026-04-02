#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PRETRAIN_ROOT}"
flowreg_python train_cfm.py "$@"
