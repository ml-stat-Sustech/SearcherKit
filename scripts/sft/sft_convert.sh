#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SFT_PYTHON="${SFT_PYTHON:-python3}"
SFT_CONVERSION_CONFIG_NAME="${SFT_CONVERSION_CONFIG_NAME:-openseeker_ms_swift}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${SFT_PYTHON}" -m searchagent plugins convert \
  --config-name "${SFT_CONVERSION_CONFIG_NAME}" \
  "$@"
