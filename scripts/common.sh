#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$(dirname "${THIS_DIR}")")" == "_project_shared_stage" ]]; then
  FLOWREG_ROOT="$(cd "${THIS_DIR}/../../.." && pwd)"
else
  FLOWREG_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
fi
PRETRAIN_ROOT="${FLOWREG_ROOT}/pretrain"
FINETUNE_ROOT="${FLOWREG_ROOT}/finetune"
SHARED_ENV_PATH="${PRETRAIN_ROOT}/.conda-env"

export TORCH_HOME="${PRETRAIN_ROOT}/.torch-cache"
export HF_HOME="${PRETRAIN_ROOT}/.hf-cache"
export PIP_CACHE_DIR="${PRETRAIN_ROOT}/.pip-cache"

mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${PIP_CACHE_DIR}"

flowreg_require_env() {
  if [[ ! -x "${SHARED_ENV_PATH}/bin/python" ]]; then
    echo "Shared env not found at ${SHARED_ENV_PATH}" >&2
    echo "Run ./scripts/setup_env.sh first." >&2
    exit 1
  fi
}

flowreg_python() {
  flowreg_require_env
  "${SHARED_ENV_PATH}/bin/python" "$@"
}

flowreg_env_tool() {
  if command -v mamba >/dev/null 2>&1; then
    echo "mamba"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return
  fi
  echo "Neither mamba nor conda is available in PATH." >&2
  exit 1
}
