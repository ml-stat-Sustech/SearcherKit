#!/usr/bin/env bash
# Run the single-round RAG agent on a dataset, optionally using the local wiki toolchain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

AGENT=rag
DATASET_NAME=${DATASET_NAME:-/mnt/sharedata/ssd_large/common/datasets/Agent/single_depth5.jsonl}
DATASET_SPLIT="${DATASET_SPLIT:-main}"
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/sharedata/hdd/beier/Agent/RAG/LocalWiki/Depth-5/Qwen2.5-32B-Instruct/rag_predictions.jsonl}
MAX_ROUNDS="${MAX_ROUNDS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
LOG_FILE="${LOG_FILE:-/mnt/sharedata/hdd/beier/Agent/RAG/LocalWiki/Depth-5/Qwen2.5-32B-Instruct/}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_REJUDGE="${FORCE_REJUDGE:-1}"
USE_SEPARATE_JUDGE_LLM="${USE_SEPARATE_JUDGE_LLM:-1}"
EVAL_OUTPUT=${EVAL_OUTPUT:-/mnt/sharedata/hdd/beier/Agent/RAG/LocalWiki/Depth-5/Qwen2.5-32B-Instruct/rag_predictions_scored.jsonl}
ENABLE_LOCAL_WIKI="${ENABLE_LOCAL_WIKI:-1}"

if [[ "${ENABLE_LOCAL_WIKI}" == "1" ]]; then
  export RAG_USE_LOCAL_WIKI=1
  export LOCAL_WIKI_INDEX="${LOCAL_WIKI_INDEX:-wiki20251001_qwen3-embedding-0.6b}"
  export LOCAL_WIKI_ES_HOST="${LOCAL_WIKI_ES_HOST:-http://192.168.77.12:9200}"
  export LOCAL_WIKI_RETRIEVER="${LOCAL_WIKI_RETRIEVER:-dense}"
  export LOCAL_WIKI_MODEL_NAME="${LOCAL_WIKI_MODEL_NAME:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B}"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cmd=(
  python -m src.webagent.main
  --agent "${AGENT}"
  --dataset-name "${DATASET_NAME}"
  --dataset-split "${DATASET_SPLIT}"
  --output-path "${OUTPUT_PATH}"
  --max-rounds "${MAX_ROUNDS}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  cmd+=(--log-file "${LOG_FILE}")
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  cmd+=(--run-eval)
  if [[ "${FORCE_REJUDGE:-0}" == "1" ]]; then
    cmd+=(--force-rejudge)
  fi
  if [[ "${USE_SEPARATE_JUDGE_LLM:-0}" == "1" ]]; then
    cmd+=(--use-separate-judge-llm)
  fi
  cmd+=(--eval-output-path "${EVAL_OUTPUT}")
fi

printf 'Running command:\n  %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
