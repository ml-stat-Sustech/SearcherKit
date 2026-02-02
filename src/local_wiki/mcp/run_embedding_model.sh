#!/bin/bash
deactivate

source /mnt/sharedata/ssd_large/users/hyli/vllm/bin/activate

export CUDA_VISIBLE_DEVICES=6
export VLLM_MODEL="${VLLM_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B}"
export VLLM_PORT="${VLLM_PORT:-8200}"

vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --max_num_batched_tokens 4 \
  --max-model-len 128 \
  --gpu-memory-utilization 0.05
  # --enforce-eager \
