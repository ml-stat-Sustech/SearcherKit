#!/usr/bin/env bash
set -euo pipefail

export SearchAgent_LOG_LEVEL=WARN
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m searchagent.training.train_dist \
    --config "${REPO_ROOT}/src/searchagent/config/training/train_dist.yaml" \
    scheduler.type=ray \
    cluster.n_nodes=2 \
    actor.backend=megatron:d3t2 \
    actor.weight_update_mode=xccl \
    actor.optimizer.lr=5e-6 \
    actor.optimizer.lr_scheduler_type=constant \
    actor.optimizer.warmup_steps_proportion=0 \
    actor.path=/home/jovyan1/Qwen3-8B \
    rollout.backend=sglang:d6 \
    rollout.max_concurrent_rollouts=256 \
    gconfig.n_samples=8 \
    +recover.no_save_optim=true \
    +recover.no_load_optim=true \
    trial_name=qwen3_new_$(date +%Y%m%d_%H%M%S) \
    train_dataset.batch_size=132 \
    train_dataset.path=/home/jovyan1/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean.jsonl
    # actor.path=/home/jovyan/agentic_sft/output/Qwen3-8B_sft_Openseeker_qwen3_yarn_262k/v4-20260327-045741/checkpoint-1459 
    # train_dataset.path=/home/jovyan1/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean.jsonl
    # no save & load optim for megatron capability
# qwen3_train_$(date +%Y%m%d_%H%M%S)
