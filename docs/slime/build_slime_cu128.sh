#!/bin/bash

set -ex

# SearchAgent local variant for the current Coder/K8s environment.
# This is intentionally a local CUDA 12.8 / conda setup, not the upstream
# Slime installation path. For a clean Slime-only deployment, prefer Slime's
# official Docker image or official CUDA 12.9 conda recipe instead of mixing
# this cu128 stack with the upstream environment.
#
# Differences from upstream build_conda.sh:
# - Reuses the already-created conda env instead of installing micromamba.
# - Uses system CUDA 12.8 instead of installing CUDA 12.9.
# - Pins the SGLang stack to v0.5.9/cu128, matching the working
#   searchagent_sglang environment on this machine.
# - Applies Slime's v0.5.9 patches to SGLang/Megatron.

ENV_NAME="${ENV_NAME:-searchagent_slime}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
  echo "Please run: conda activate $ENV_NAME" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$BASE_DIR/third_party}"
SLIME_DIR="${SLIME_DIR:-$THIRD_PARTY_DIR/slime}"
SGLANG_DIR="${SGLANG_DIR:-$THIRD_PARTY_DIR/sglang}"
MEGATRON_DIR="${MEGATRON_DIR:-$THIRD_PARTY_DIR/Megatron-LM}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS="${MAX_JOBS:-16}"
export INSTALL_FLASH_ATTN_3="${INSTALL_FLASH_ATTN_3:-1}"
export INSTALL_TRANSFORMER_ENGINE="${INSTALL_TRANSFORMER_ENGINE:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TRANSFORMER_ENGINE_VERSION="${TRANSFORMER_ENGINE_VERSION:-2.10.0}"

export SGLANG_VERSION="${SGLANG_VERSION:-v0.5.9}"
export SLIME_REF="${SLIME_REF:-a73a1496634f78f499ae399666c5c92853c653ff}"
export MEGATRON_COMMIT="${MEGATRON_COMMIT:-1dcf0dafa884ad52ffb243625717a3471643e087}"
export PATCH_VERSION="${PATCH_VERSION:-v0.5.9}"

python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY
nvcc --version

# Match the working searchagent_sglang environment.
uv pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install "sglang==0.5.9" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match

pip install cmake ninja "setuptools<80.0.0" pybind11 "packaging>=24.2"

# Slime itself also carries the SGLang/Megatron patch files and conversion scripts.
mkdir -p "$THIRD_PARTY_DIR"
if [ ! -d "$SLIME_DIR/.git" ]; then
  git clone https://github.com/THUDM/slime.git "$SLIME_DIR"
fi
cd "$SLIME_DIR"
git fetch origin "$SLIME_REF"
git checkout "$SLIME_REF"

# Install editable SGLang source and apply the Slime patch for v0.5.9.
if [ ! -d "$SGLANG_DIR/.git" ]; then
  git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
fi
cd "$SGLANG_DIR"
git fetch --tags
git checkout "$SGLANG_VERSION"
git update-index --refresh || true
if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" 2>/dev/null; then
  git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "sglang patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "sglang patch already applied or not applicable, skipping"
fi
uv pip install -e "python" --no-deps

# Native training dependencies used by Slime/Megatron.
# Keep FA2: Megatron/TransformerEngine still use the flash_attn package path.
MAX_JOBS="$MAX_JOBS" pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# Hopper FA3 is a separate package path (flash_attn_3). H20/Hopper + CUDA 12.8
# can use it, and Slime's Dockerfile installs it from the hopper/ subdir.
if [ "$INSTALL_FLASH_ATTN_3" = "1" ]; then
  FLASH_ATTN_DIR="${FLASH_ATTN_DIR:-$THIRD_PARTY_DIR/flash-attention}"
  if [ ! -d "$FLASH_ATTN_DIR/.git" ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_DIR"
  fi
  cd "$FLASH_ATTN_DIR"
  git fetch origin fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89
  git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89
  git submodule update --init
  cd hopper
  MAX_JOBS="$MAX_JOBS" python setup.py install
  python_path="$(python -c 'import site; print(site.getsitepackages()[0])')"
  mkdir -p "$python_path/flash_attn_3"
  cp flash_attn_interface.py "$python_path/flash_attn_3/flash_attn_interface.py"
fi

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install flash-linear-attention==0.4.1
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
if [ "$INSTALL_TRANSFORMER_ENGINE" = "1" ]; then
  SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
  CUDA_INCLUDE_DIRS="$(find "$SITE_PACKAGES/nvidia" -maxdepth 3 -type d -name include | sort | paste -sd: -)"
  CUDA_LIB_DIRS="$(find "$SITE_PACKAGES/nvidia" -maxdepth 3 -type d -name lib | sort | paste -sd: -)"
  export CPATH="$CUDA_INCLUDE_DIRS${CPATH:+:$CPATH}"
  export C_INCLUDE_PATH="$CUDA_INCLUDE_DIRS${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE_DIRS${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export LIBRARY_PATH="$CUDA_LIB_DIRS${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$CUDA_LIB_DIRS:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export NVTE_FRAMEWORK=pytorch

  pip install --no-cache-dir \
    "transformer_engine==${TRANSFORMER_ENGINE_VERSION}" \
    "transformer_engine_cu12==${TRANSFORMER_ENGINE_VERSION}" \
    onnx onnxscript onnx_ir importlib_metadata zipp
  pip install --no-cache-dir --no-build-isolation --no-deps \
    "transformer_engine_torch==${TRANSFORMER_ENGINE_VERSION}"
else
  echo "Skipping TransformerEngine build. Set INSTALL_TRANSFORMER_ENGINE=1 if Megatron requires it."
fi

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
pip install -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation
pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# Megatron-LM.
if [ ! -d "$MEGATRON_DIR/.git" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive "$MEGATRON_DIR"
fi
cd "$MEGATRON_DIR"
git checkout "$MEGATRON_COMMIT"
git update-index --refresh || true
if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" 2>/dev/null; then
  git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "megatron patch already applied or not applicable, skipping"
fi
pip install -e . --no-build-isolation

# Slime itself.
cd "$SLIME_DIR"
pip install -r requirements.txt
pip install -e . --no-deps

cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install . --no-build-isolation

# Keep Megatron happy. Re-check SGLang after this because numpy is downgraded.
pip install "numpy<2"

python - <<'PY'
import importlib.metadata as md
import torch
import sglang
import sglang_router
import slime
import megatron

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("sglang:", md.version("sglang"), sglang.__file__)
print("sglang_router:", sglang_router.__version__)
print("slime:", slime.__file__)
print("megatron:", megatron.__file__)
PY
