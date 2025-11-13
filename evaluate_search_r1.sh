#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: evaluate_search_r1.sh <predictions_jsonl> [output_jsonl] [judge_dataset] [judge_model] [judge_prompt_file]

Arguments:
  predictions_jsonl  Path to the Search-R1 rollout results (e.g. .../iter1.jsonl).
  output_jsonl       Destination for judged results (default: <predictions_basename>_judge.jsonl).
  judge_dataset      Dataset key for builtin prompt defaults (default: webwalker).
  judge_model        Override judge model name (optional; otherwise OpenAI env vars are used).
  judge_prompt_file  Optional custom prompt file; contents will be passed to --judge_prompt.

Environment variables of interest:
  OPENAI_JUDGE_MODEL, OPENAI_JUDGE_API_KEY, OPENAI_JUDGE_MODEL_SERVER
    Set these to point to your custom LLM-as-a-judge endpoint when --judge_model is omitted.
  OPENAI_MODEL / OPENAI_API_KEY
    Fallback judge credentials if *_JUDGE_* are not provided.

Example:
  ./evaluate_search_r1.sh outputs/search_r1/<dataset>/iter1.jsonl \
      outputs/search_r1/<dataset>/iter1_judge.jsonl \
      webwalker \
      judge-model-name \
      prompts/my_custom_judge_prompt.txt
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

INPUT_PATH=$1
shift

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Error: predictions file '$INPUT_PATH' does not exist." >&2
  exit 1
fi

DEFAULT_OUTPUT="${INPUT_PATH%.*}_judge.jsonl"
OUTPUT_PATH=${1:-$DEFAULT_OUTPUT}
if [[ $# -ge 1 ]]; then shift; fi

JUDGE_DATASET=${1:-webwalker}
if [[ $# -ge 1 ]]; then shift; fi

JUDGE_MODEL=${1:-}
if [[ $# -ge 1 ]]; then shift; fi

JUDGE_PROMPT=""
if [[ $# -ge 1 ]]; then
  PROMPT_FILE=$1
  shift
  if [[ -f "$PROMPT_FILE" ]]; then
    JUDGE_PROMPT=$(<"$PROMPT_FILE")
  else
    echo "Warning: judge prompt file '$PROMPT_FILE' not found; ignoring." >&2
  fi
fi

PYTHON_CMD=("python" "src/webagent/evaluate/evl.py" "--input_path" "$INPUT_PATH" "--output_path" "$OUTPUT_PATH" "--judge_dataset" "$JUDGE_DATASET")

if [[ -n "$JUDGE_MODEL" ]]; then
  PYTHON_CMD+=("--judge_model" "$JUDGE_MODEL")
fi

if [[ -n "$JUDGE_PROMPT" ]]; then
  PYTHON_CMD+=("--judge_prompt" "$JUDGE_PROMPT")
fi

echo "Running LLM judge evaluation..."
"${PYTHON_CMD[@]}"
