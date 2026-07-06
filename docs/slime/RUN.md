# SearchAgent Slime Run

This document describes how to run the current two-machine SearchAgent Slime setup after completing `docs/slime/INSTALL.md`.

## Topology

```text
Training machine: local 8 GPU
  GPU 0-3: Megatron actor training
  GPU 4-7: SGLang rollout

Service machine: 10.32.33.135
  Elasticsearch: 10.32.33.135:9200
  GPU 0: embedding / TEI
  GPU 1-7: summary / SGLang
```

The service machine does not join Ray. The training machine starts a local Ray head only.

## Repository Config

`src/searchagent/config/training/train_slime.yaml` should point both Elasticsearch sources to the service machine:

```yaml
hosts: ["http://10.32.33.135:9200"]
embedding_base_url: http://10.32.33.135:8004/v1
summary_base_url: http://10.32.33.135:6010/v1
```

Current service-side concurrency:

```yaml
embedding_max_concurrency: 128
summary_max_concurrency: 28
```

`scripts/slime/train_grpo.sh` defaults to the local 4+4 GPU split:

```bash
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE=4
ROLLOUT_NUM_GPUS=4
ROLLOUT_NUM_GPUS_PER_ENGINE=4
NUM_GPUS_PER_NODE=8
```

Algorithm switching uses:

```bash
ADVANTAGE_ESTIMATOR=grpo
ADVANTAGE_ESTIMATOR=igpo
```

## Service Machine

### Elasticsearch

From the training machine:

```bash
curl http://10.32.33.135:9200
```

If this is unavailable, fix Elasticsearch or networking before changing training parameters.

### Embedding

On `10.32.33.135`, start TEI on GPU 0:

```bash
cd /home/jovyan1/wsy/searchagent/searchagent-slime/docs/slime/embedding_tools

bash start_embedding.sh \
  --frontend-port 8004 \
  --backend-start 8020 \
  --gpus 0 \
  --model /home/jovyan/Qwen3-Embedding-8B
```

Single-GPU mode listens directly on `8004` and does not require nginx. Nginx is only needed when serving multiple TEI backends.
The helper writes local runtime files under `docs/slime/embedding_tools/tei_logs/` and `tei_pids.txt`; these are runtime artifacts and should not be committed.

Check locally on the service machine:

```bash
curl http://127.0.0.1:8004/health
curl http://127.0.0.1:8004/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/jovyan/Qwen3-Embedding-8B","input":"hello"}'
```

Check from the training machine:

```bash
curl http://10.32.33.135:8004/health
curl http://10.32.33.135:8004/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/jovyan/Qwen3-Embedding-8B","input":"hello"}'
```

### Summary

On `10.32.33.135`, start summary SGLang on GPU 1-7:

```bash
conda activate searchagent_sglang

QWEN3_MODEL=/home/jovyan/Qwen3-8B

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m sglang.launch_server \
  --model-path "$QWEN3_MODEL" \
  --host 0.0.0.0 \
  --port 6010 \
  --served-model-name "$QWEN3_MODEL" \
  --data-parallel-size 7 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --context-length 40960 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --reasoning-parser qwen3
```

Use `tmux` or an equivalent process supervisor for this long-running service.

Check from the training machine:

```bash
curl http://10.32.33.135:6010/v1/models

curl http://10.32.33.135:6010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/home/jovyan/Qwen3-8B",
    "messages": [
      {
        "role": "user",
        "content": "Return exactly this JSON object: {\"evidence\":\"hello\", \"summary\":\"hello\"}"
      }
    ],
    "max_tokens": 512,
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Expected: `choices[0].message.content` is not `null`. If output appears mainly in `reasoning_content`, thinking is not disabled.

## Training Machine

Start Ray from the `searchagent_slime` environment:

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

ray stop --force
ray start --head \
  --node-ip-address 10.32.32.6 \
  --port 6379 \
  --dashboard-host 0.0.0.0 \
  --dashboard-port 8265 \
  --num-gpus 8 \
  --disable-usage-stats
```

Check:

```bash
ray status
```

Expected: one active node with `8.0 GPU`.

## Standalone Tests

Run these before starting a full GRPO/IGPO job. They isolate environment, external services, and SearchAgent tool wiring.

### Runtime Imports

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

export PYTHONPATH="${PWD}/third_party/Megatron-LM:${PWD}/third_party/slime:${PWD}/third_party/sglang/python:${PWD}/src:${PWD}:${PYTHONPATH}"

python - <<'PY'
import searchagent
import slime
import sglang
import megatron.training

print("searchagent:", searchagent.__file__)
print("slime:", slime.__file__)
print("sglang:", sglang.__file__)
print("megatron:", megatron.training.__file__)
PY
```

### Training CLI

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

PYTHONPATH="${PWD}/third_party/Megatron-LM:${PWD}/third_party/slime:${PWD}/third_party/sglang/python:${PWD}/src:${PWD}:${PYTHONPATH}" \
python -m searchagent.training.slime.train_async --help | head -80
```

This should show the full Slime/Megatron CLI, not only the fallback SearchAgent-specific help.

### Search Services

```bash
curl http://10.32.33.135:9200
curl http://10.32.33.135:8004/health
curl http://10.32.33.135:6010/v1/models
```

Embedding request:

```bash
curl http://10.32.33.135:8004/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/jovyan/Qwen3-Embedding-8B","input":"hello"}'
```

Summary request:

```bash
curl http://10.32.33.135:6010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/home/jovyan/Qwen3-8B",
    "messages": [
      {
        "role": "user",
        "content": "Return exactly this JSON object: {\"evidence\":\"hello\", \"summary\":\"hello\"}"
      }
    ],
    "max_tokens": 512,
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### SearchAgent Tool Wiring

This tests `src/searchagent/config/training/train_slime.yaml` without launching Slime training. It builds the configured eval search tool and performs one search through Elasticsearch, embedding, and summary.

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}" python - <<'PY'
import asyncio
from omegaconf import OmegaConf

from searchagent.sources import add_source_cfg
from searchagent.tools import build_tool
from searchagent.training.config import AgentConfig


async def main():
    raw = OmegaConf.load("src/searchagent/config/training/train_slime.yaml")
    cfg = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(AgentConfig), raw.eval_agent)
    )
    for source_cfg in cfg.sources:
        add_source_cfg(source_cfg.name, source_cfg)
    tool = build_tool(cfg.tools[0])
    await tool.init()
    try:
        result = await tool.call({"query": "BrowseComp Plus benchmark"})
        print(str(result)[:2000])
    finally:
        await tool.close()


asyncio.run(main())
PY
```

Expected: a non-empty search result/summary. If this fails, fix SearchAgent service config before starting training.

### Minimal Slime Smoke

This launches Ray actors and runs one rollout. Use it after the standalone service tests pass:

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

export RAY_ADDRESS=auto

TRIAL_NAME=qwen3_slime_smoke_$(date +%Y%m%d_%H%M%S) \
ADVANTAGE_ESTIMATOR=grpo \
bash scripts/slime/train_grpo.sh \
  --num-rollout 1 \
  --save-interval 1 \
  --skip-eval-before-train
```

## Start Training

GRPO:

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

export RAY_ADDRESS=auto
mkdir -p logs/slime/run_logs

RUN_TS=$(date +%Y%m%d_%H%M%S)
TRIAL_NAME=qwen3_slime_grpo_${RUN_TS} \
ADVANTAGE_ESTIMATOR=grpo \
bash scripts/slime/train_grpo.sh 2>&1 | tee logs/slime/run_logs/grpo_${RUN_TS}.log
```

IGPO:

```bash
conda activate searchagent_slime
cd /home/jovyan1/wsy/searchagent/searchagent-slime

export RAY_ADDRESS=auto
mkdir -p logs/slime/run_logs

RUN_TS=$(date +%Y%m%d_%H%M%S)
TRIAL_NAME=qwen3_slime_igpo_${RUN_TS} \
bash scripts/slime/train_igpo.sh 2>&1 | tee logs/slime/run_logs/igpo_${RUN_TS}.log
```

Short smoke:

```bash
TRIAL_NAME=qwen3_slime_smoke_$(date +%Y%m%d_%H%M%S) \
ADVANTAGE_ESTIMATOR=grpo \
bash scripts/slime/train_grpo.sh \
  --num-rollout 1 \
  --save-interval 1 \
  --skip-eval-before-train
```

## Current Training Defaults

```text
actor training:              1 node x 4 GPU
rollout:                     4 GPU, one SGLang engine
TP/CP/PP:                    TP=2, CP=2, PP=1
rollout max context len:     40960
eval max context len:        40960
rollout max response len:    36864
eval max response len:       24576
SGLang mem fraction static:  0.7
rollout batch size:          128 prompts
n samples per prompt:        8
num steps per rollout:       1
global batch size:           1024 samples
total train steps:           about 630 for the current 13448-line train set
learning rate:               5e-6, override with LR=...
eval interval:               every 32 train steps, plus epoch boundaries
eval agent max_turn:         40
eval agent timeout:          1800 seconds, wrap-up prompt at 300 seconds remaining
advantage estimator:         grpo by default, set ADVANTAGE_ESTIMATOR=igpo for IGPO
```

## Config Checks

Confirm the new service IP is configured:

```bash
rg -n "10\\.32\\.33\\.135|10\\.32\\.34\\.135|10\\.32\\.29\\.237|embedding_max_concurrency|summary_max_concurrency" \
  src/searchagent/config/training/train_slime.yaml
```

Expected: only `10.32.33.135`, no old `10.32.34.135` or `10.32.29.237`.

Confirm the local 4+4 training split:

```bash
rg -n "ACTOR_NUM_GPUS_PER_NODE|ROLLOUT_NUM_GPUS|ROLLOUT_NUM_GPUS_PER_ENGINE|ROLLOUT_BATCH_SIZE|NUM_STEPS_PER_ROLLOUT|EVAL_INTERVAL|ADVANTAGE_ESTIMATOR" \
  scripts/slime/train_grpo.sh
```

## Troubleshooting

`summary content is null`: ensure the summary request/config disables thinking with `chat_template_kwargs.enable_thinking=false`.

`embedding OOM or high latency`: lower `embedding_max_concurrency` in `src/searchagent/config/training/train_slime.yaml`, for example from `128` to `64`.

`summary connection error`: check `curl http://10.32.33.135:6010/v1/models`, then verify the SGLang summary process is still alive on the service machine.

`40960 context error`: keep rollout, eval, agent, and summary context settings at `40960` for Qwen3-8B.

`Ray sees extra GPUs or extra nodes`: restart Ray with `ray stop --force`, then start only the local head for this topology.

`ActorDiedError ... Server process terminated unexpectedly`: search earlier SGLang server logs for the first real traceback, especially context length, CUDA/NVCC, import path, and OOM errors.

`No module named 'megatron.training'`: Ray was started without the `third_party/Megatron-LM` path. Export the environment variables from `docs/slime/INSTALL.md`, restart Ray, then rerun training.
