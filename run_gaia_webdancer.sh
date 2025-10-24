#!/usr/bin/env bash
# Run WebDancer on the GAIA dataset with optional local wiki tooling.
#
# Environment knobs (override as needed):
#   DATASET_NAME          Defaults to "./GAIA"
#   DATASET_SPLIT         Defaults to "validation"
#   OUTPUT_PATH           Defaults to "<repo>/runs/gaia_webdancer.jsonl"
#   MAX_ROUNDS            Defaults to 20
#   MAX_SAMPLES           Optional limit on processed samples
#   LOG_FILE              Optional path for verbose run logs
#   RUN_EVAL              Defaults to 1 (set to 0 to skip LLM judge evaluation)
#
# Local wiki (local_wiki) configuration:
#   USE_LOCAL_WIKI        Defaults to 1 (enable local wiki flags)
#   LOCAL_WIKI_INDEX      Name of the Elasticsearch index
#   LOCAL_WIKI_ES_HOST    Elasticsearch endpoint (http://127.0.0.1:9200 by default)
#   LOCAL_WIKI_ES_TIMEOUT Optional request timeout in seconds
#   LOCAL_WIKI_RETRIEVER  bm25 | dense | hybrid
#   LOCAL_WIKI_MODEL_NAME Required when retriever is dense/hybrid
#
# Judge overrides (only used when RUN_EVAL=1):
#   JUDGE_DATASET, JUDGE_MODEL, JUDGE_PROMPT, FORCE_REJUDGE (set to 1), USE_SEPARATE_JUDGE_LLM (set to 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

USE_SEPARATE_JUDGE_LLM=1
DATASET_NAME=/mnt/sharedata/ssd_large/common/datasets/GAIA
DATASET_SPLIT="${DATASET_SPLIT:-validation}"
OUTPUT_PATH="${OUTPUT_PATH:-$/mnt/sharedata/hdd/beier/Agent/WebDancer/gaia_webdancer.jsonl}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
LOG_FILE="${LOG_FILE:-}"
RUN_EVAL="${RUN_EVAL:-1}"
JUDGE_DATASET="${JUDGE_DATASET:-webwalker}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cmd=(
  python -m src.webagent.main
  --agent webdancer
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
  if [[ -n "${JUDGE_DATASET}" ]]; then
    cmd+=(--judge-dataset "${JUDGE_DATASET}")
  fi
  if [[ -n "${JUDGE_MODEL:-}" ]]; then
    cmd+=(--judge-model "${JUDGE_MODEL}")
  fi
  if [[ -n "${JUDGE_PROMPT:-}" ]]; then
    cmd+=(--judge-prompt "${JUDGE_PROMPT}")
  fi
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

if [[ "${USE_SEPARATE_JUDGE_LLM:-0}" == "1" ]]; then
  ARGS+=(--use-separate-judge-llm)
fi

printf 'Running command:\n  %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
