#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CELOSS_PARALLEL_SIZE="${CELOSS_PARALLEL_SIZE:-2048}"
export REPORT_TO="${REPORT_TO:-swanlab}"

SFT_DEFAULT_CUDA_HOME="${SFT_DEFAULT_CUDA_HOME:-/data/rubrics/verl/.cuda-toolkit}"
if [[ -z "${CUDA_HOME:-}" && -x "${SFT_DEFAULT_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${SFT_DEFAULT_CUDA_HOME}"
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
    NVCC_BIN="$(command -v nvcc || true)"
    if [[ -n "${NVCC_BIN}" ]]; then
        export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
    fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
    export PATH="${PATH}:${CUDA_HOME}/bin"
    export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SFT_PYTHON="${SFT_PYTHON:-}"
if [[ -z "${SFT_PYTHON}" && -x "/home/jovyan/miniconda3/envs/webagent-sft/bin/python3" ]]; then
    SFT_PYTHON="/home/jovyan/miniconda3/envs/webagent-sft/bin/python3"
fi
SFT_PYTHON="${SFT_PYTHON:-python3}"
SFT_TRAINING_CONFIG="${SFT_TRAINING_CONFIG:-${SCRIPT_DIR}/sft_training.yaml}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/kjob/torch_extensions}"

PY_SITE="$("${SFT_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
CURAND_LIB_DIR="${PY_SITE}/nvidia/curand/lib"
SFT_LIB_DIR="${REPO_ROOT}/kjob/lib"
if [[ -f "${CURAND_LIB_DIR}/libcurand.so.10" ]]; then
    mkdir -p "${SFT_LIB_DIR}"
    ln -sf "${CURAND_LIB_DIR}/libcurand.so.10" "${SFT_LIB_DIR}/libcurand.so"
    export LIBRARY_PATH="${SFT_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${SFT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

CUDA_WHEEL_LIBS=(
    "${PY_SITE}/torch/lib"
    "${PY_SITE}/nvidia/cuda_runtime/lib"
    "${PY_SITE}/nvidia/cuda_nvrtc/lib"
    "${PY_SITE}/nvidia/cublas/lib"
    "${CURAND_LIB_DIR}"
)
for LIB_DIR in "${CUDA_WHEEL_LIBS[@]}"; do
    if [[ -d "${LIB_DIR}" ]]; then
        export LD_LIBRARY_PATH="${LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
if [[ -d "${PY_SITE}/nvidia/curand/include" ]]; then
    export CPATH="${PY_SITE}/nvidia/curand/include${CPATH:+:${CPATH}}"
    export CPLUS_INCLUDE_PATH="${PY_SITE}/nvidia/curand/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
fi
if [[ -d "${CURAND_LIB_DIR}" ]]; then
    export LIBRARY_PATH="${CURAND_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

"${SFT_PYTHON}" -m searcherkit.training.sft --config "${SFT_TRAINING_CONFIG}" "$@"
