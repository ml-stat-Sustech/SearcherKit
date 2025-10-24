#!/usr/bin/env bash
set -euo pipefail

AGENT=webwalker
USE_SEPARATE_JUDGE_LLM=1
DATASET_NAME=/mnt/sharedata/ssd_large/common/datasets/WebWalkerQA
DATASET_SPLIT=main
OUTPUT_PATH=/mnt/sharedata/hdd/beier/Agent/WebWalker/webwalker_predictions_demo.jsonl
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_ROUNDS=10
LOG_FILE=/mnt/sharedata/hdd/beier/Agent/logs/
EVAL_OUTPUT=${EVAL_OUTPUT:-/mnt/sharedata/hdd/beier/Agent/WebWalker/webwalker_predictions_scored_demp.jsonl}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
mkdir -p "$(dirname "${EVAL_OUTPUT}")"

ARGS=(
  --agent "${AGENT}"
  --dataset-name "${DATASET_NAME}"
  --dataset-split "${DATASET_SPLIT}"
  --output-path "${OUTPUT_PATH}"
  --run-eval
  --eval-output-path "${EVAL_OUTPUT}"
  --force-rejudge
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

if [[ "${USE_SEPARATE_JUDGE_LLM:-0}" == "1" ]]; then
  ARGS+=(--use-separate-judge-llm)
fi

python -m src.webagent.main "${ARGS[@]}" "$@"
