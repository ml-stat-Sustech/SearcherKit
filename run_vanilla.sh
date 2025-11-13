#!/usr/bin/env bash
# Run the Vanilla agent over a dataset with optional evaluation.

set -euo pipefail

AGENT=vanilla
DATASET_NAME=${DATASET_NAME:-/mnt/sharedata/ssd_large/common/datasets/Agent/walker_multi_10.jsonl}
DATASET_SPLIT=${DATASET_SPLIT:-validation}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/sharedata/hdd/beier/Agent/Vanilla/LocalWiki/walker_multi_10/DeepResearch-30B/webdancer_results.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_ROUNDS=${MAX_ROUNDS:-4}
LOG_FILE=${LOG_FILE:-}
RUN_EVAL=${RUN_EVAL:-1}
EVAL_OUTPUT=${EVAL_OUTPUT:-/mnt/sharedata/hdd/beier/Agent/Vanilla/DeepResearch-30B/vanilla_predictions_eval.jsonl}
FORCE_REJUDGE=${FORCE_REJUDGE:-1}
USE_SEPARATE_JUDGE_LLM=${USE_SEPARATE_JUDGE_LLM:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
if [[ "${RUN_EVAL}" == "1" ]]; then
  mkdir -p "$(dirname "${EVAL_OUTPUT}")"
fi

ARGS=(
  --agent "${AGENT}"
  --dataset-name "${DATASET_NAME}"
  --dataset-split "${DATASET_SPLIT}"
  --output-path "${OUTPUT_PATH}"
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

if [[ "${RUN_EVAL}" == "1" ]]; then
  ARGS+=(--run-eval --eval-output-path "${EVAL_OUTPUT}")
  if [[ "${FORCE_REJUDGE}" == "1" ]]; then
    ARGS+=(--force-rejudge)
  fi
  if [[ "${USE_SEPARATE_JUDGE_LLM}" == "1" ]]; then
    ARGS+=(--use-separate-judge-llm)
  fi
fi

python -m src.webagent.main "${ARGS[@]}" "$@"
