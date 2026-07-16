#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SFT_PYTHON="${SFT_PYTHON:-python3}"
SFT_CONVERSION_INPUT="${SFT_CONVERSION_INPUT:-/data/hf/hub/agent_tmp/agentic_sft/data/openseeker_red_search_visit_agentic.jsonl}"
SFT_TRAIN_DATASET="${SFT_TRAIN_DATASET:-outputs/sft/openseeker-qwen3-8b/ms_swift.jsonl}"
SFT_CONVERSION_MAX_RECORDS="${SFT_CONVERSION_MAX_RECORDS:-0}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${SFT_PYTHON}" -m searcherkit plugins convert \
  "${SFT_CONVERSION_INPUT}" \
  "${SFT_TRAIN_DATASET}" \
  --max-records "${SFT_CONVERSION_MAX_RECORDS}" \
  "$@"
