#!/usr/bin/env bash
# Run WebDancer across multiple single_depth datasets and collect accuracies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

DATASETS=(
  "/mnt/sharedata/ssd_large/common/datasets/Agent/walker_multi_10.jsonl"
)

OUTPUT_ROOT="/mnt/sharedata/hdd/beier/Agent/WebDancer/LocalWiki/"
MODEL_TAG="${MODEL_TAG:-DeepResearch-30B}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DATASET_SPLIT="${DATASET_SPLIT:-validation}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_REJUDGE="${FORCE_REJUDGE:-1}"
USE_SEPARATE_JUDGE_LLM="${USE_SEPARATE_JUDGE_LLM:-1}"
USE_LOCAL_WIKI="${USE_LOCAL_WIKI:-1}"
CSV_PATH="${CSV_PATH:-${OUTPUT_ROOT}/${MODEL_TAG}/webdancer_accuracy.csv}"
DISABLE_ACTIONABLE_LINKS="${DISABLE_ACTIONABLE_LINKS:-1}"

if [[ -n "${DISABLE_ACTIONABLE_LINKS}" ]] && [[ -z "${LOCAL_WIKI_DISABLE_LINKS:-}" ]]; then
  export LOCAL_WIKI_DISABLE_LINKS="${DISABLE_ACTIONABLE_LINKS}"
fi

mkdir -p "$(dirname "${CSV_PATH}")"
printf 'dataset,label,predictions,eval,report,accuracy\n' > "${CSV_PATH}"

for dataset in "${DATASETS[@]}"; do
  base="$(basename "${dataset}")"
  label="${base%.jsonl}"
  run_dir="${OUTPUT_ROOT}/${label}/${MODEL_TAG}"
  mkdir -p "${run_dir}"

  output_path="${run_dir}/predictions.jsonl"
  eval_output="${run_dir}/predictions_eval.jsonl"
  log_file="${run_dir}/run.log"

  cmd=(
    python -m src.webagent.main
    --agent webdancer
    --dataset-name "${dataset}"
    --dataset-split "${DATASET_SPLIT}"
    --output-path "${output_path}"
    --eval-output-path "${eval_output}"
    --max-rounds "${MAX_ROUNDS}"
    --log-file "${log_file}"
  )

  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    cmd+=(--run-eval)
    if [[ "${FORCE_REJUDGE}" == "1" ]]; then
      cmd+=(--force-rejudge)
    fi
    if [[ "${USE_SEPARATE_JUDGE_LLM}" == "1" ]]; then
      cmd+=(--use-separate-judge-llm)
    fi
  fi

  if [[ "${USE_LOCAL_WIKI}" == "1" ]]; then
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

  printf '[webdancer-batch] Running %s\n' "${label}"
  "${cmd[@]}"

  report_path="${eval_output%.*}_report.json"
  accuracy=""
  if [[ -f "${report_path}" ]]; then
    accuracy="$(
      python -c '
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data.get("overall")
    if isinstance(overall, (int, float)):
        print(f"{overall:.4f}")
except Exception:
    pass
' "${report_path}"
    )"
  fi

  printf '%s,%s,%s,%s,%s,%s\n' \
    "${dataset}" \
    "${label}" \
    "${output_path}" \
    "${eval_output}" \
    "${report_path}" \
    "${accuracy}" >> "${CSV_PATH}"
done

printf '[webdancer-batch] Accuracy summary saved to %s\n' "${CSV_PATH}"
