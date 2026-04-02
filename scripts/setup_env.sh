#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_TOOL="$(flowreg_env_tool)"

if [[ -d "${SHARED_ENV_PATH}" ]]; then
  "${ENV_TOOL}" env update -p "${SHARED_ENV_PATH}" -f "${PRETRAIN_ROOT}/environment.yml"
else
  "${ENV_TOOL}" env create -p "${SHARED_ENV_PATH}" -f "${PRETRAIN_ROOT}/environment.yml"
fi

"${ENV_TOOL}" env update -p "${SHARED_ENV_PATH}" -f "${FINETUNE_ROOT}/environment.yml"

echo "Shared env ready at ${SHARED_ENV_PATH}"
