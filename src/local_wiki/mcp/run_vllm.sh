#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$(cd "${SCRIPT_DIR}/../.."; pwd)"

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B}"
export VLLM_PORT="${VLLM_PORT:-8200}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen3-Embedding-0.6B}"

echo "Starting vLLM with model: $VLLM_MODEL_PATH"
echo "Port: $VLLM_PORT"

vllm serve "$VLLM_MODEL_PATH" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --embedding-mode \
  --trust-remote-code