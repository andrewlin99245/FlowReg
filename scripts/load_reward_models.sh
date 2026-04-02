#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${FLOWREG_ROOT}"
flowreg_python scripts/bootstrap_reward_models.py "$@"
