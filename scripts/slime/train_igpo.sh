#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export SEARCHERKIT_AGENT_CONFIG="${SEARCHERKIT_AGENT_CONFIG:-${REPO_ROOT}/src/searcherkit/config/training/train_slime.yaml}"

cd "${REPO_ROOT}"

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-igpo}"

if [ -n "${IGPO_MODE:-}" ]; then
    case "${IGPO_MODE}" in
        sync)
            if [ -n "${SLIME_ASYNC_MODE:-}" ] && [ "${SLIME_ASYNC_MODE}" != "0" ]; then
                echo "IGPO_MODE=sync conflicts with SLIME_ASYNC_MODE=${SLIME_ASYNC_MODE}" >&2
                exit 2
            fi
            export SLIME_ASYNC_MODE=0
            ;;
        async)
            if [ -n "${SLIME_ASYNC_MODE:-}" ] && [ "${SLIME_ASYNC_MODE}" != "1" ]; then
                echo "IGPO_MODE=async conflicts with SLIME_ASYNC_MODE=${SLIME_ASYNC_MODE}" >&2
                exit 2
            fi
            export SLIME_ASYNC_MODE=1
            ;;
        *)
            echo "Unsupported IGPO_MODE=${IGPO_MODE}; expected sync or async" >&2
            exit 2
            ;;
    esac
else
    # Default to the AReal-like async IGPO path. Set IGPO_MODE=sync or
    # SLIME_ASYNC_MODE=0 explicitly for a synchronous sanity run.
    export SLIME_ASYNC_MODE="${SLIME_ASYNC_MODE:-1}"
    if [ "${SLIME_ASYNC_MODE}" = "1" ]; then
        IGPO_MODE=async
    elif [ "${SLIME_ASYNC_MODE}" = "0" ]; then
        IGPO_MODE=sync
    else
        echo "Unsupported SLIME_ASYNC_MODE=${SLIME_ASYNC_MODE}; expected 0 or 1" >&2
        exit 2
    fi
fi

export TRIAL_NAME="${TRIAL_NAME:-qwen3_slime_igpo_${IGPO_MODE}_${RUN_TS}}"

# IGPO reward mixture. Keep these aligned with the local Slime GRPO baseline
# unless explicitly overridden for an ablation.
export SEARCHERKIT_IGPO_REWARD_COEF="${SEARCHERKIT_IGPO_REWARD_COEF:-1.0}"
export SEARCHERKIT_IGPO_OUTCOME_REWARD_COEF="${SEARCHERKIT_IGPO_OUTCOME_REWARD_COEF:-1.0}"
export SEARCHERKIT_IGPO_REWARD_SIDE="${SEARCHERKIT_IGPO_REWARD_SIDE:-actor}"
export SEARCHERKIT_IGPO_ACTOR_SCORE_MICRO_BATCH_SIZE="${SEARCHERKIT_IGPO_ACTOR_SCORE_MICRO_BATCH_SIZE:-8}"
export SEARCHERKIT_TRUNCATION_PENALTY="${SEARCHERKIT_TRUNCATION_PENALTY:--1.0}"
export DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-searcherkit.training.slime.rollout.areal_outcome_reward_filter}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-searcherkit.training.slime.rollout.areal_outcome_reward_post_process}"

# Common comparison defaults inherited by train_grpo.sh. They are repeated here
# so the IGPO entry point is self-documenting while still avoiding script drift.
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-4}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
if [ "${IGPO_MODE}" = "async" ]; then
    # Keep AReal's behavior-ratio correction as token-level clamp via Slime TIS.
    # The PPO ratio can be switched to contiguous valid-token step spans to match
    export SEARCHERKIT_PPO_RATIO_MODE="${SEARCHERKIT_PPO_RATIO_MODE:-step}"
    export USE_TIS="${USE_TIS:-1}"
    export TIS_CLIP="${TIS_CLIP:-5.0}"
else
    export SEARCHERKIT_PPO_RATIO_MODE="${SEARCHERKIT_PPO_RATIO_MODE:-token}"
    export USE_TIS="${USE_TIS:-0}"
    export TIS_CLIP="${TIS_CLIP:-5.0}"
fi
export TIS_CLIP_LOW="${TIS_CLIP_LOW:-0}"

mkdir -p logs/slime/run_logs
if [ ! -f "${SEARCHERKIT_AGENT_CONFIG}" ]; then
    echo "SearcherKit config not found: ${SEARCHERKIT_AGENT_CONFIG}" >&2
    exit 2
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'IGPO_MODE=%s\n' "${IGPO_MODE}"
    printf 'SLIME_ASYNC_MODE=%s\n' "${SLIME_ASYNC_MODE}"
    printf 'USE_TIS=%s\n' "${USE_TIS}"
    printf 'TIS_CLIP=%s\n' "${TIS_CLIP}"
    printf 'TIS_CLIP_LOW=%s\n' "${TIS_CLIP_LOW}"
    printf 'N_SAMPLES_PER_PROMPT=%s\n' "${N_SAMPLES_PER_PROMPT}"
    printf 'NUM_STEPS_PER_ROLLOUT=%s\n' "${NUM_STEPS_PER_ROLLOUT}"
    printf 'KL_LOSS_COEF=%s\n' "${KL_LOSS_COEF}"
    printf 'SEARCHERKIT_IGPO_REWARD_SIDE=%s\n' "${SEARCHERKIT_IGPO_REWARD_SIDE}"
    printf 'SEARCHERKIT_IGPO_ACTOR_SCORE_MICRO_BATCH_SIZE=%s\n' "${SEARCHERKIT_IGPO_ACTOR_SCORE_MICRO_BATCH_SIZE}"
    printf 'SEARCHERKIT_PPO_RATIO_MODE=%s\n' "${SEARCHERKIT_PPO_RATIO_MODE}"
    printf 'DYNAMIC_SAMPLING_FILTER_PATH=%s\n' "${DYNAMIC_SAMPLING_FILTER_PATH}"
    printf 'CUSTOM_REWARD_POST_PROCESS_PATH=%s\n' "${CUSTOM_REWARD_POST_PROCESS_PATH}"
    printf 'SEARCHERKIT_AGENT_CONFIG=%s\n' "${SEARCHERKIT_AGENT_CONFIG}"
    printf 'TRIAL_NAME=%s\n' "${TRIAL_NAME}"
    printf 'LOG_PATH=%s\n' "logs/slime/run_logs/igpo_${IGPO_MODE}_${RUN_TS}.log"
    exit 0
fi

bash "${SCRIPT_DIR}/train_grpo.sh" "$@" 2>&1 | tee "logs/slime/run_logs/igpo_${IGPO_MODE}_${RUN_TS}.log"
