#!/usr/bin/env bash
set -euo pipefail

export SEARCHERKIT_LOG_LEVEL=WARN
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SEARCHERKIT_AGENT_CONFIG="${SEARCHERKIT_AGENT_CONFIG:-${REPO_ROOT}/src/searcherkit/config/training/train_slime.yaml}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-${REPO_ROOT}/third_party}"
prepend_pythonpath_if_dir() {
    local path="$1"
    if [ -d "${path}" ]; then
        export PYTHONPATH="${path}:${PYTHONPATH}"
    fi
}
prepend_pythonpath_if_dir "${REPO_ROOT}/Megatron-LM"
prepend_pythonpath_if_dir "${REPO_ROOT}/slime"
prepend_pythonpath_if_dir "${REPO_ROOT}/sglang/python"
prepend_pythonpath_if_dir "${THIRD_PARTY_DIR}/Megatron-LM"
prepend_pythonpath_if_dir "${THIRD_PARTY_DIR}/slime"
prepend_pythonpath_if_dir "${THIRD_PARTY_DIR}/sglang/python"

if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        NVCC_BIN="$(command -v nvcc)"
        export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME=/usr/local/cuda
    else
        for CUDA_DIR in /usr/local/cuda-* /usr/local/cuda; do
            if [ -x "${CUDA_DIR}/bin/nvcc" ]; then
                export CUDA_HOME="${CUDA_DIR}"
                break
            fi
        done
    fi
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
CUDA_LIB_DIRS=""
if [ -d "${SITE_PACKAGES}/nvidia" ]; then
    CUDA_LIB_DIRS="$(find "${SITE_PACKAGES}/nvidia" -maxdepth 3 -type d -name lib | sort | paste -sd: -)"
fi
LD_LIBRARY_PARTS="${CUDA_LIB_DIRS}"
if [ -n "${CONDA_PREFIX:-}" ]; then
    LD_LIBRARY_PARTS="${LD_LIBRARY_PARTS:+${LD_LIBRARY_PARTS}:}${CONDA_PREFIX}/lib"
fi
if [ -n "${LD_LIBRARY_PARTS}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PARTS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [ -n "${CUDA_LIB_DIRS}" ]; then
    export LIBRARY_PATH="${CUDA_LIB_DIRS}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-/home/jovyan1/Qwen3-8B}"
MEGATRON_CKPT="${MEGATRON_CKPT:-/home/jovyan1/Qwen3-8B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/home/jovyan1/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean.jsonl}"
VALID_DATA="${VALID_DATA:-/home/jovyan1/browsecomp_plus_decrypted_qa.jsonl}"
SAVE_ROOT="${SAVE_ROOT:-/home/jovyan1/wsy/searcherkit/searcherkit-slime/logs/slime}"
TRIAL_NAME="${TRIAL_NAME:-qwen3_slime_$(date +%Y%m%d_%H%M%S)}"
ACTOR_LOAD="${ACTOR_LOAD:-${SAVE_ROOT}/${TRIAL_NAME}}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-webagent}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-${SAVE_ROOT}/swanlab}"
SWANLAB_GROUP="${SWANLAB_GROUP:-${TRIAL_NAME}}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"
SWANLAB_ARGS=(
    --use-swanlab
    --swanlab-mode "${SWANLAB_MODE}"
    --swanlab-project "${SWANLAB_PROJECT}"
    --swanlab-experiment-name "${TRIAL_NAME}"
    --swanlab-group "${SWANLAB_GROUP}"
    --swanlab-logdir "${SWANLAB_LOGDIR}"
)
if [ -n "${SWANLAB_WORKSPACE:-}" ]; then
    SWANLAB_ARGS+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
fi
if [ -n "${SWANLAB_HOST:-}" ]; then
    SWANLAB_ARGS+=(--swanlab-host "${SWANLAB_HOST}")
fi
if [ -n "${SWANLAB_API_HOST:-}" ]; then
    SWANLAB_ARGS+=(--swanlab-host "${SWANLAB_API_HOST}")
fi
if [ -n "${SWANLAB_WEB_HOST:-}" ]; then
    SWANLAB_ARGS+=(--swanlab-web-host "${SWANLAB_WEB_HOST}")
fi
EVAL_BEFORE_TRAIN_ARGS=()
if [ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]; then
    EVAL_BEFORE_TRAIN_ARGS+=(--skip-eval-before-train)
fi
TRAIN_ENV_VARS="${TRAIN_ENV_VARS:-$(python3 -c 'import json, os; print(json.dumps({"CUDA_DEVICE_MAX_CONNECTIONS":"1","SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN":"1","PYTHONPATH":os.environ.get("PYTHONPATH", ""),"CUDA_HOME":os.environ.get("CUDA_HOME", ""),"PATH":os.environ.get("PATH", ""),"LD_LIBRARY_PATH":os.environ.get("LD_LIBRARY_PATH", ""),"LIBRARY_PATH":os.environ.get("LIBRARY_PATH", "")}))')}"

# 8-GPU default: 4 GPUs for Megatron actor training, 4 GPUs for SGLang rollout.
# External search services live on 10.32.33.135. Override these env vars for
# larger-node or multi-node runs.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-64}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
EVAL_INTERVAL="${EVAL_INTERVAL:-32}"
# Save on the same periodic cadence as eval by default; slime also triggers
# periodic actions at epoch boundaries.
SAVE_INTERVAL="${SAVE_INTERVAL:-${EVAL_INTERVAL}}"
SLIME_ASYNC_MODE="${SLIME_ASYNC_MODE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
# Match AReal's ppo_n_minibatches=4: split one rollout batch into 4
# optimizer updates.
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-36864}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-24576}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-40960}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-${ROLLOUT_MAX_CONTEXT_LEN}}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-2}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((ROLLOUT_MAX_CONTEXT_LEN / CONTEXT_PARALLEL_SIZE))}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"
LR="${LR:-5e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SEARCHERKIT_EVAL_CONCURRENCY="${SEARCHERKIT_EVAL_CONCURRENCY:-64}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
SEARCHERKIT_IGPO_REWARD_COEF="${SEARCHERKIT_IGPO_REWARD_COEF:-1.0}"
SEARCHERKIT_IGPO_OUTCOME_REWARD_COEF="${SEARCHERKIT_IGPO_OUTCOME_REWARD_COEF:-1.0}"
SEARCHERKIT_IGPO_REWARD_SIDE="${SEARCHERKIT_IGPO_REWARD_SIDE:-rollout}"
SEARCHERKIT_IGPO_ACTOR_SCORE_MICRO_BATCH_SIZE="${SEARCHERKIT_IGPO_ACTOR_SCORE_MICRO_BATCH_SIZE:-8}"
SEARCHERKIT_PPO_RATIO_MODE="${SEARCHERKIT_PPO_RATIO_MODE:-token}"
SEARCHERKIT_TRUNCATION_PENALTY="${SEARCHERKIT_TRUNCATION_PENALTY:--1.0}"
DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-searcherkit.training.slime.rollout.mixed_reward_filter}"
CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
USE_TIS="${USE_TIS:-${SLIME_ASYNC_MODE}}"
TIS_CLIP="${TIS_CLIP:-2.0}"
TIS_CLIP_LOW="${TIS_CLIP_LOW:-0}"
CUSTOM_TIS_FUNCTION_PATH="${CUSTOM_TIS_FUNCTION_PATH:-}"
if [ "${ADVANTAGE_ESTIMATOR}" != "grpo" ] && [ "${ADVANTAGE_ESTIMATOR}" != "igpo" ]; then
    echo "Unsupported ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR}; expected grpo or igpo" >&2
    exit 2
fi
TIS_ARGS=()
if [ "${USE_TIS}" = "1" ]; then
    TIS_ARGS+=(--use-tis --tis-clip "${TIS_CLIP}" --tis-clip-low "${TIS_CLIP_LOW}")
fi
if [ -n "${CUSTOM_TIS_FUNCTION_PATH}" ]; then
    TIS_ARGS+=(--custom-tis-function-path "${CUSTOM_TIS_FUNCTION_PATH}")
fi
ADVANTAGE_ARGS=(--advantage-estimator "${ADVANTAGE_ESTIMATOR}")
if [ "${ADVANTAGE_ESTIMATOR}" = "igpo" ]; then
    ADVANTAGE_ARGS+=(--custom-advantage-function-path searcherkit.training.slime.igpo.compute_advantages_and_returns)
fi
CUSTOM_REWARD_POST_PROCESS_ARGS=()
if [ -n "${CUSTOM_REWARD_POST_PROCESS_PATH}" ]; then
    CUSTOM_REWARD_POST_PROCESS_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi
if [ ! -f "${SEARCHERKIT_AGENT_CONFIG}" ]; then
    echo "SearcherKit config not found: ${SEARCHERKIT_AGENT_CONFIG}" >&2
    exit 2
fi

if [ "${SLIME_ASYNC_MODE}" = "1" ]; then
    SLIME_TRAIN_MODULE="${SLIME_TRAIN_MODULE:-searcherkit.training.slime.train_async}"
    ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-searcherkit.training.slime.fully_async.generate_rollout_fully_async}"
else
    SLIME_TRAIN_MODULE="${SLIME_TRAIN_MODULE:-searcherkit.training.slime.train_dist}"
    ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.sglang_rollout.generate_rollout}"
fi
EVAL_FUNCTION_PATH="${EVAL_FUNCTION_PATH:-slime.rollout.sglang_rollout.generate_rollout}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'SLIME_TRAIN_MODULE=%s\n' "${SLIME_TRAIN_MODULE}"
    printf 'ROLLOUT_FUNCTION_PATH=%s\n' "${ROLLOUT_FUNCTION_PATH}"
    printf 'EVAL_FUNCTION_PATH=%s\n' "${EVAL_FUNCTION_PATH}"
    printf 'SLIME_ASYNC_MODE=%s\n' "${SLIME_ASYNC_MODE}"
    printf 'ADVANTAGE_ESTIMATOR=%s\n' "${ADVANTAGE_ESTIMATOR}"
    printf 'ROLLOUT_BATCH_SIZE=%s\n' "${ROLLOUT_BATCH_SIZE}"
    printf 'N_SAMPLES_PER_PROMPT=%s\n' "${N_SAMPLES_PER_PROMPT}"
    printf 'NUM_STEPS_PER_ROLLOUT=%s\n' "${NUM_STEPS_PER_ROLLOUT}"
    printf 'GLOBAL_BATCH_SIZE=%s\n' "${GLOBAL_BATCH_SIZE}"
    printf 'EVAL_INTERVAL=%s\n' "${EVAL_INTERVAL}"
    printf 'SAVE_INTERVAL=%s\n' "${SAVE_INTERVAL}"
    printf 'LR=%s\n' "${LR}"
    printf 'KL_LOSS_COEF=%s\n' "${KL_LOSS_COEF}"
    printf 'USE_TIS=%s\n' "${USE_TIS}"
    printf 'TIS_CLIP=%s\n' "${TIS_CLIP}"
    printf 'TIS_CLIP_LOW=%s\n' "${TIS_CLIP_LOW}"
    printf 'CUSTOM_TIS_FUNCTION_PATH=%s\n' "${CUSTOM_TIS_FUNCTION_PATH}"
    printf 'SEARCHERKIT_IGPO_REWARD_SIDE=%s\n' "${SEARCHERKIT_IGPO_REWARD_SIDE}"
    printf 'SEARCHERKIT_PPO_RATIO_MODE=%s\n' "${SEARCHERKIT_PPO_RATIO_MODE}"
    printf 'DYNAMIC_SAMPLING_FILTER_PATH=%s\n' "${DYNAMIC_SAMPLING_FILTER_PATH}"
    printf 'CUSTOM_REWARD_POST_PROCESS_PATH=%s\n' "${CUSTOM_REWARD_POST_PROCESS_PATH}"
    printf 'SEARCHERKIT_AGENT_CONFIG=%s\n' "${SEARCHERKIT_AGENT_CONFIG}"
    printf 'TRIAL_NAME=%s\n' "${TRIAL_NAME}"
    exit 0
fi

python3 -m "${SLIME_TRAIN_MODULE}" \
    --swiglu \
    --num-layers 36 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --rotary-base 1000000 \
    --vocab-size 151936 \
    --kv-channels 128 \
    --qk-layernorm \
    --untie-embeddings-and-output-weights \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${MEGATRON_CKPT}" \
    --load "${ACTOR_LOAD}" \
    --save "${SAVE_ROOT}/${TRIAL_NAME}" \
    "${SWANLAB_ARGS[@]}" \
    --prompt-data "${PROMPT_DATA}" \
    --input-key question \
    --label-key answer \
    --metadata-key metadata \
    --eval-prompt-data browsecomp_plus "${VALID_DATA}" \
    --eval-input-key question \
    --eval-label-key answer \
    --n-samples-per-eval-prompt 1 \
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
    --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}" \
    --eval-temperature 1.0 \
    --eval-top-p 1.0 \
    --eval-function-path "${EVAL_FUNCTION_PATH}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --searcherkit-eval-concurrency "${SEARCHERKIT_EVAL_CONCURRENCY}" \
    "${EVAL_BEFORE_TRAIN_ARGS[@]}" \
    --rollout-function-path "${ROLLOUT_FUNCTION_PATH}" \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
    --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
    --train-env-vars "${TRAIN_ENV_VARS}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --num-epoch 6 \
    --seed 1 \
    --rollout-seed 1 \
    --rollout-shuffle \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}" \
    --rollout-temperature 1.0 \
    --rollout-top-p 1.0 \
    --lr "${LR}" \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --kl-coef 0.0 \
    --use-kl-loss \
    --kl-loss-coef "${KL_LOSS_COEF}" \
    --kl-loss-type low_var_kl \
    --normalize-advantages \
    "${TIS_ARGS[@]}" \
    --optimizer adam \
    --update-weight-mode full \
    --update-weight-transport nccl \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --bf16 \
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}" \
    --sglang-router-request-timeout-secs 7200 \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    --sglang-context-length "${ROLLOUT_MAX_CONTEXT_LEN}" \
    --sglang-max-prefill-tokens "${ROLLOUT_MAX_CONTEXT_LEN}" \
    --save-interval "${SAVE_INTERVAL}" \
    "${ADVANTAGE_ARGS[@]}" \
    --balance-data \
    --custom-generate-function-path searcherkit.training.slime.rollout.generate_searcherkit \
    --custom-rm-path searcherkit.training.slime.rollout.custom_rm \
    "${CUSTOM_REWARD_POST_PROCESS_ARGS[@]}" \
    --custom-rollout-log-function-path searcherkit.training.slime.rollout.searcherkit_rollout_log \
    --custom-eval-rollout-log-function-path searcherkit.training.slime.rollout.searcherkit_eval_rollout_log \
    --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}" \
    --searcherkit-agent-config "${SEARCHERKIT_AGENT_CONFIG}" \
    --searcherkit-agent-config-key agent \
    --searcherkit-eval-agent-config-key eval_agent \
    --searcherkit-igpo-reward-coef "${SEARCHERKIT_IGPO_REWARD_COEF}" \
    --searcherkit-igpo-outcome-reward-coef "${SEARCHERKIT_IGPO_OUTCOME_REWARD_COEF}" \
    --searcherkit-igpo-reward-side "${SEARCHERKIT_IGPO_REWARD_SIDE}" \
    --searcherkit-igpo-actor-score-micro-batch-size "${SEARCHERKIT_IGPO_ACTOR_SCORE_MICRO_BATCH_SIZE}" \
    --searcherkit-ppo-ratio-mode "${SEARCHERKIT_PPO_RATIO_MODE}" \
    --searcherkit-truncation-penalty "${SEARCHERKIT_TRUNCATION_PENALTY}" \
    "$@"
