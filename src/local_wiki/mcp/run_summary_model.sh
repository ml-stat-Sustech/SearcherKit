#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$(cd "${SCRIPT_DIR}/../.."; pwd)"

export VLLM_MODEL_PATH="${SUMMARY_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-8B}"
export VLLM_PORT="${SUMMARY_MODEL_PORT:-8300}"
export VLLM_MODEL_NAME="${SUMMARY_MODEL_NAME:-Qwen3-8B}"

echo "Starting vLLM summary model: $VLLM_MODEL_PATH"
echo "Port: $VLLM_PORT"

vllm serve "$VLLM_MODEL_PATH" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --dtype float8 \
  --gpu-memory-utilization 0.7 \
  --max_num_batched_tokens 8192
