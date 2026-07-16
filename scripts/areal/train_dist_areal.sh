#!/usr/bin/env bash
set -euo pipefail

export SEARCHERKIT_LOG_LEVEL=WARN
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m searcherkit.training.areal.train_dist \
    --config "${REPO_ROOT}/src/searcherkit/config/training/train_areal.yaml" \
    scheduler.type=ray \
    cluster.n_nodes=2 \
    actor.backend=fsdp:d4c2 \
    actor.weight_update_mode=xccl \
    actor.optimizer.lr=5e-6 \
    actor.optimizer.lr_scheduler_type=constant \
    actor.optimizer.warmup_steps_proportion=0 \
    actor.path=/home/jovyan1/Qwen3-8B \
    rollout.backend=sglang:d8 \
    rollout.max_concurrent_rollouts=256 \
    gconfig.n_samples=8 \
    trial_name=qwen3_new_$(date +%Y%m%d_%H%M%S) \
    train_dataset.batch_size=128 \
    train_dataset.path=/home/jovyan1/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean.jsonl \
    "$@"
