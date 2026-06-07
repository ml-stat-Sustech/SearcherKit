#!/usr/bin/env bash
set -euo pipefail

export SearchAgent_LOG_LEVEL=WARN
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/home/jovyan1/Qwen3-8B}"
PROMPT_DATA="${PROMPT_DATA:-/home/jovyan1/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean.jsonl}"
VALID_DATA="${VALID_DATA:-/home/jovyan1/browsecomp_plus_decrypted_qa.jsonl}"
SAVE_ROOT="${SAVE_ROOT:-/home/jovyan1/searchagent/outputs/slime}"
TRIAL_NAME="${TRIAL_NAME:-qwen3_slime_$(date +%Y%m%d_%H%M%S)}"

python3 -m searchagent.training.train_dist_slime \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${HF_CHECKPOINT}" \
    --save "${SAVE_ROOT}/${TRIAL_NAME}" \
    --prompt-data "${PROMPT_DATA}" \
    --input-key question \
    --label-key answer \
    --metadata-key metadata \
    --eval-prompt-data browsecomp_plus "${VALID_DATA}" \
    --eval-input-key question \
    --eval-label-key answer \
    --n-samples-per-eval-prompt 1 \
    --eval-max-response-len 8192 \
    --eval-max-context-len 65536 \
    --eval-temperature 1.0 \
    --eval-top-p 1.0 \
    --eval-interval 32 \
    --actor-num-nodes 2 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 8 \
    --num-gpus-per-node 8 \
    --rollout-batch-size 128 \
    --n-samples-per-prompt 8 \
    --num-epoch 6 \
    --seed 1 \
    --rollout-seed 1 \
    --rollout-shuffle \
    --global-batch-size 256 \
    --num-steps-per-rollout 4 \
    --micro-batch-size 1 \
    --rollout-max-response-len 8192 \
    --rollout-max-context-len 65536 \
    --rollout-temperature 1.0 \
    --rollout-top-p 1.0 \
    --lr 5e-6 \
    --clip-grad 1.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --kl-coef 0.0 \
    --normalize-advantages \
    --save-interval 32 \
    --advantage-estimator grpo \
    --custom-generate-function-path searchagent.training.slime_rollout.generate_searchagent \
    --custom-rm-path searchagent.training.slime_rollout.custom_rm \
    --dynamic-sampling-filter-path searchagent.training.slime_rollout.mixed_reward_filter \
    --searchagent-agent-config "${SCRIPT_DIR}/searchagent_slime_agent.yaml" \
    --searchagent-agent-config-key agent \
    --searchagent-eval-agent-config-key eval_agent \
    "$@"
