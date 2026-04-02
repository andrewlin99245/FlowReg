#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_TOOL="$(flowreg_env_tool)"
USE_VERIFIED_LOCK=0
VERIFIED_LOCK_PATH="${PRETRAIN_ROOT}/environment.verified.lock.yml"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_env.sh [--verified-lock]

Options:
  --verified-lock  Use the exported exact dev-env snapshot at
                   pretrain/environment.verified.lock.yml. This is best for
                   matching the verified local environment on a similar
                   platform. The default path uses the pinned shared env files
                   and is the safer option for CUDA deployment on a different
                   machine.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verified-lock)
      USE_VERIFIED_LOCK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${USE_VERIFIED_LOCK}" -eq 1 ]]; then
  if [[ ! -f "${VERIFIED_LOCK_PATH}" ]]; then
    echo "Verified lock file not found at ${VERIFIED_LOCK_PATH}" >&2
    exit 1
  fi
  if [[ -d "${SHARED_ENV_PATH}" ]]; then
    "${ENV_TOOL}" env update -p "${SHARED_ENV_PATH}" -f "${VERIFIED_LOCK_PATH}" --prune
  else
    "${ENV_TOOL}" env create -p "${SHARED_ENV_PATH}" -f "${VERIFIED_LOCK_PATH}"
  fi
  echo "Shared env ready at ${SHARED_ENV_PATH} using ${VERIFIED_LOCK_PATH}"
  exit 0
fi

if [[ -d "${SHARED_ENV_PATH}" ]]; then
  "${ENV_TOOL}" env update -p "${SHARED_ENV_PATH}" -f "${PRETRAIN_ROOT}/environment.yml"
else
  "${ENV_TOOL}" env create -p "${SHARED_ENV_PATH}" -f "${PRETRAIN_ROOT}/environment.yml"
fi

"${ENV_TOOL}" env update -p "${SHARED_ENV_PATH}" -f "${FINETUNE_ROOT}/environment.yml"

echo "Shared env ready at ${SHARED_ENV_PATH} using pinned shared env files"
