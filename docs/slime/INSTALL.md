# SearchAgent Slime Install

This document describes the local SearchAgent Slime environment used in the current Coder/K8s workspace.

The install path here is intentionally a local CUDA 12.8 conda setup. It is not the upstream Slime install path. For a clean Slime-only deployment, prefer Slime's official Docker image or official CUDA 12.9 conda recipe. Do not mix the official cu129/Docker dependency stack with this local cu128 environment.

## Scope

Current repository:

```text
/home/jovyan1/wsy/searchagent/searchagent-slime
```

Third-party source trees are kept under:

```text
third_party/slime
third_party/sglang
third_party/Megatron-LM
third_party/flash-attention
```

The local build installs patched editable source trees because Slime training needs SGLang rollout weight-update hooks and Megatron actor/checkpoint compatibility patches.

On a new machine, `docs/slime/build_slime_cu128.sh` creates `third_party/` automatically, clones Slime/SGLang/Megatron-LM/flash-attention into it, checks out the pinned revisions, applies Slime's SGLang/Megatron patches, and installs the editable packages. You do not need to copy the local `third_party/` directory by hand.

## Base Conda Env

```bash
conda create -n searchagent_slime python=3.12 -y
conda activate searchagent_slime

pip install -U pip uv ninja cmake packaging setuptools wheel
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

The local build script installs the cu128 PyTorch stack and Slime runtime dependencies:

```bash
cd /home/jovyan1/wsy/searchagent/searchagent-slime
bash docs/slime/build_slime_cu128.sh
```

The script installs:

- PyTorch cu128
- patched editable SGLang from `third_party/sglang`
- pinned Slime from `third_party/slime`
- patched editable Megatron-LM from `third_party/Megatron-LM`
- SGLang router
- Megatron-Bridge / mbridge
- flash-attn / optional FA3
- TransformerEngine
- Slime native kernels

## SearchAgent Package

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

uv pip install -e ".[elasticsearch-source]" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
```

If `uv` is unavailable:

```bash
pip install -e ".[elasticsearch-source]"
```

## Environment Variables

Use these before Ray startup, model conversion, and manual import checks:

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

export CUDA_DEVICE_MAX_CONNECTIONS=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="${PWD}/third_party/Megatron-LM:${PWD}/third_party/slime:${PWD}/third_party/sglang/python:${PWD}/src:${PWD}:${PYTHONPATH}"

SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
CUDA_LIB_DIRS="$(find "$SITE_PACKAGES/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | sort | paste -sd: -)"
export LD_LIBRARY_PATH="$CUDA_LIB_DIRS:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_LIB_DIRS${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

`scripts/slime/train_grpo.sh` also prepares these paths for the training process, but Ray itself should be started from a shell that already has them.

## Import Check

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

export PYTHONPATH="${PWD}/third_party/Megatron-LM:${PWD}/third_party/slime:${PWD}/third_party/sglang/python:${PWD}/src:${PWD}:${PYTHONPATH}"

python - <<'PY'
import torch
import searchagent
import slime
import sglang
import sglang_router
import megatron.training
import transformer_engine.pytorch as te
import transformer_engine_torch

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("searchagent:", searchagent.__file__)
print("slime:", slime.__file__)
print("sglang:", getattr(sglang, "__version__", None), sglang.__file__)
print("sglang_router:", getattr(sglang_router, "__version__", None))
print("megatron:", megatron.training.__file__)
print("te:", te.__file__)
print("te_torch:", transformer_engine_torch.__file__)
PY
```

CLI/help check:

```bash
PYTHONPATH="${PWD}/third_party/Megatron-LM:${PWD}/third_party/slime:${PWD}/third_party/sglang/python:${PWD}/src:${PWD}:${PYTHONPATH}" \
python -m searchagent.training.slime.train_async --help | head -80
```

## Model Conversion

Slime actor training uses a Megatron distributed checkpoint. Convert the Hugging Face checkpoint once:

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime/third_party/slime

source scripts/models/qwen3-8B.sh

PYTHONPATH=/home/jovyan1/wsy/searchagent/searchagent-slime/third_party/Megatron-LM \
python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /home/jovyan1/Qwen3-8B \
  --save /home/jovyan1/Qwen3-8B_torch_dist
```

Check the output:

```bash
cat /home/jovyan1/Qwen3-8B_torch_dist/latest_checkpointed_iteration.txt
find /home/jovyan1/Qwen3-8B_torch_dist -maxdepth 2 -type f | head
```

Expected `latest_checkpointed_iteration.txt`:

```text
release
```

## Common Install Issues

`No module named 'megatron.training'`: `third_party/Megatron-LM` is missing from `PYTHONPATH`.

`libcudnn_graph.so.9 not found`: export the `LD_LIBRARY_PATH` built from pip-installed NVIDIA libraries, then restart Ray.

`apply_rope_fusion is not available`: TransformerEngine native extension is missing or mismatched. Re-run `docs/slime/build_slime_cu128.sh` in `searchagent_slime`.

`flash_attn_3 seems to be not installed`: only required if the model/runtime explicitly enables FA3. The current Slime training path does not require FA3 for the Qwen3-8B setup.

`deep_ep is not installed`: acceptable for the current dense Qwen3-8B run; MoE/DeepEP paths need additional setup.
