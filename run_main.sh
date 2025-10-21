#!/usr/bin/env bash
set -euo pipefail

AGENT=webwalker
DATASET_NAME=/mnt/sharedata/ssd_large/common/datasets/WebWalkerQA
DATASET_SPLIT=main
OUTPUT_PATH=/mnt/sharedata/hdd/beier/Agent/WebWalker/webwalker_predictions.jsonl
MAX_SAMPLES=1
MAX_ROUNDS=30
LLM_BACKEND=${LLM_BACKEND:-auto}
LOG_FILE=/mnt/sharedata/hdd/beier/Agent/logs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

ARGS=(
  --agent "${AGENT}"
  --dataset-name "${DATASET_NAME}"
  --dataset-split "${DATASET_SPLIT}"
  --output-path "${OUTPUT_PATH}"
  --llm "${LLM_BACKEND}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${MAX_ROUNDS}" ]]; then
  ARGS+=(--max-rounds "${MAX_ROUNDS}")
fi

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  ARGS+=(--log-file "${LOG_FILE}")
fi

exec python -m src.main "${ARGS[@]}" "$@"
