#!/usr/bin/env bash
# Run WebDancer on the GAIA dataset with optional local wiki tooling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

# Local wiki summary model configuration (edit values or export before running)
export LOCAL_WIKI_SUMMARY_MODEL="${LOCAL_WIKI_SUMMARY_MODEL:-qwen/qwen-2.5-72b-instruct}"
export LOCAL_WIKI_SUMMARY_API_KEY="${LOCAL_WIKI_SUMMARY_API_KEY:-REMOVED_REVOKED_SECRET}"
export LOCAL_WIKI_SUMMARY_BASE_URL="${LOCAL_WIKI_SUMMARY_BASE_URL:-https://openrouter.ai/api/v1}"

AGENT=webdancer
USE_SEPARATE_JUDGE_LLM=1
DATASET_NAME=/mnt/sharedata/ssd_large/common/datasets/Agent/WebSailor.json
DATASET_SPLIT="${DATASET_SPLIT:-validation}"
OUTPUT_PATH=/mnt/sharedata/hdd/beier/Agent/WebDancer/LocalWiki/WebSailor/Qwen2.5-32B-Instruct/webdancer_results_test.jsonl
MAX_ROUNDS="${MAX_ROUNDS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
LOG_FILE="${LOG_FILE:-/mnt/sharedata/hdd/beier/Agent/WebDancer/LocalWiki/WebSailor/Qwen2.5-32B-Instruct/}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_REJUDGE="${FORCE_REJUDGE:-1}"
EVAL_OUTPUT=${EVAL_OUTPUT:-/mnt/sharedata/hdd/beier/Agent/WebDancer/LocalWiki/WebSailor/Qwen2.5-32B-Instruct/webdancer_predictions_scored_test.jsonl}

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
fi

if [[ "${USE_LOCAL_WIKI:-1}" == "1" ]]; then
  cmd+=(--use-local-wiki-tools)
  if [[ -n "${LOCAL_WIKI_INDEX:-}" ]]; then
    cmd+=(--local-wiki-index "${LOCAL_WIKI_INDEX}")
  fi
  if [[ -n "${LOCAL_WIKI_ES_HOST:-}" ]]; then
    cmd+=(--local-wiki-es-host "${LOCAL_WIKI_ES_HOST}")
  fi
  if [[ -n "${LOCAL_WIKI_ES_TIMEOUT:-}" ]]; then
    cmd+=(--local-wiki-es-timeout "${LOCAL_WIKI_ES_TIMEOUT}")
  fi
  if [[ -n "${LOCAL_WIKI_RETRIEVER:-}" ]]; then
    cmd+=(--local-wiki-retriever "${LOCAL_WIKI_RETRIEVER}")
  fi
  if [[ -n "${LOCAL_WIKI_MODEL_NAME:-}" ]]; then
    cmd+=(--local-wiki-model-name "${LOCAL_WIKI_MODEL_NAME}")
  fi
fi

printf 'Running command:\n  %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
