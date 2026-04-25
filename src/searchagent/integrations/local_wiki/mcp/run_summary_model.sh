#!/bin/bash
deactivate

source /mnt/sharedata/ssd_large/users/hyli/vllm/bin/activate

export CUDA_VISIBLE_DEVICES=0

export VLLM_MODEL="${SUMMARY_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-8B-FP8}"

python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port "${SUMMARY_MODEL_PORT:-8300}" \
  --dtype auto \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max_num_batched_tokens 8192 \
  --max-model-len 32768 \
  # --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
