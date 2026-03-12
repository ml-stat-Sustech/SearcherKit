#!/usr/bin/env bash
# Run WebDancer on the GAIA dataset with optional local wiki tooling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:$(cd "${REPO_ROOT}/.."; pwd)"

USE_LOCAL_WIKI=0
# Local wiki summary model configuration (edit values or export before running)
# export LOCAL_WIKI_SUMMARY_MODEL="${LOCAL_WIKI_SUMMARY_MODEL:-Qwen/Qwen3-8B}"
# export LOCAL_WIKI_SUMMARY_API_KEY="${LOCAL_WIKI_SUMMARY_API_KEY:-sk-iuseapibjpdeahdvmxquvitungchbbtzhqsjhkhlfrpwevef}"
# export LOCAL_WIKI_SUMMARY_BASE_URL="${LOCAL_WIKI_SUMMARY_BASE_URL:-https://api.siliconflow.cn/v1}"

export OPENAI_MODEL='/mnt/sharedata/ssd_large/common/LLMs/WebSailor-7B'
export OPENAI_API_KEY="a"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
# export OPENAI_TEMPERATURE=0.6

# export OPENAI_MODEL='/mnt/sharedata/ssd_large/common/LLMs/WebSailor-7B'
# export OPENAI_BASE_URL="http://192.168.77.12:8000/v1"
# export OPENAI_API_KEY="1"

export OPENAI_JUDGE_MODEL='qwen3-32b'
export OPENAI_JUDGE_API_KEY="a"
export OPENAI_JUDGE_MODEL_SERVER="https://www.dmxapi.cn/v1"

AGENT=webdancer
USE_SEPARATE_JUDGE_LLM=1
DATASET_NAME=../agent_data/browsecomp_plus_decrypted_simple.jsonl
DATASET_SPLIT="${DATASET_SPLIT:-train}"
OUTPUT_PATH=./results/websailor_browsecomp_plus_simple_results.jsonl
MAX_ROUNDS="${MAX_ROUNDS:-30}"
# MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_SAMPLES="100"
LOG_FILE=./log
RUN_EVAL=1
FORCE_REJUDGE=1
EVAL_OUTPUT=./results/websailor_browsecomp_plus_simple_predictions_scored_test.jsonl

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
