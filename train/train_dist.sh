#!/usr/bin/env bash
set -euo pipefail

export SearchAgent_LOG_LEVEL="${SearchAgent_LOG_LEVEL:-WARN}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"
export CUDA_HOME="${CUDA_HOME:-/data/rubrics/verl/.cuda-toolkit}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CURAND_INCLUDE="${CURAND_INCLUDE:-/home/jovyan/miniconda3/envs/webagent-train/lib/python3.12/site-packages/nvidia/curand/include}"
export CURAND_LIB="${CURAND_LIB:-/home/jovyan/miniconda3/envs/webagent-train/lib/python3.12/site-packages/nvidia/curand/lib}"
export CPATH="${CURAND_INCLUDE}${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CURAND_INCLUDE}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${CURAND_LIB}:${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CURAND_LIB}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRIAL_NAME="${TRIAL_NAME:-qwen3_searchagent_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${REPO_ROOT}/outputs/areal/experiments" "${REPO_ROOT}/outputs/areal/name_resolve"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m searchagent.training.train_dist \
    --config "${SCRIPT_DIR}/train_dist.yaml" \
    trial_name="${TRIAL_NAME}" \
    "$@"
