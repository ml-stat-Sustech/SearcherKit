# SearcherKit Training Launch

本文记录当前三台 workspace 的启动流程和已验证状态。

## 当前状态

- `search-agent-04` 已启动 embedding server。
  - 当前 pod IP: `10.32.20.11`
  - Endpoint: `http://10.32.20.11:8004/v1`
  - 已从 `search-agent-02` 验证 `/v1/models` 和 `/v1/embeddings` 可用。
- `search-agent-04` 已启动 summary server。
  - 当前 pod IP: `10.32.20.11`
  - Endpoint: `http://10.32.20.11:6010/v1`
  - 已从 `search-agent-02` 验证 `/v1/models` 和 `/v1/chat/completions` 可用。
  - `response_format={"type":"json_object"}` 和 `chat_template_kwargs.enable_thinking=false` 已验证生效。
- 训练日志已配置为 SwanLab cloud。
  - SwanLab project: `searcherkit`
  - SwanLab experiment: `zero_query_repeat_align`
  - WandB 已关闭。
- `train/train_dist.sh` 已内置 CUDA/curand 环境变量，用于 FlashInfer sampling JIT 编译。
  - 已单独重跑 `/home/jovyan/.cache/flashinfer/0.6.7.post3/90a/cached_ops/sampling` 的 ninja 编译，当前可以通过。
- `search-agent-02` 和 `search-agent-03` 的 Ray 集群已启动。
  - Head: `search-agent-02`, IP `10.32.56.200`, port `6379`
  - Worker: `search-agent-03`
  - `ray status` 显示 2 个 active nodes、`16.0 GPU`、无 pending、无 failures。

注意：`search-agent-04` 这个 hostname 在 02 内部无法解析，当前 `train/train_dist.yaml` 已直接写入 `10.32.20.11`。如果 04 workspace 重启，pod IP 可能改变，需要在 02 上重新执行：

```bash
kubectl get pods -o wide
```

## 公共路径

```bash
REPO=/home/jovyan/code/searcherkit
QWEN3_MODEL=/data/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
EMB_MODEL=/data/hf/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
```

## 04 Embedding Server

在 `search-agent-04` 上用 `tmux new -s embed` 启动：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate webagent-sglang

EMB_MODEL=/data/hf/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af

CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path "$EMB_MODEL" \
  --host 0.0.0.0 \
  --port 8004 \
  --served-model-name "$EMB_MODEL" \
  --is-embedding \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --trust-remote-code
```

看到 `Application startup complete` 后，按 `Ctrl-b d` detach。

## 04 Summary Server

在 `search-agent-04` 上用 `tmux new -s summary` 启动：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate webagent-sglang

QWEN3_MODEL=/data/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m sglang.launch_server \
  --model-path "$QWEN3_MODEL" \
  --host 0.0.0.0 \
  --port 6010 \
  --served-model-name "$QWEN3_MODEL" \
  --data-parallel-size 6 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --context-length 40960 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --reasoning-parser qwen3
```

看到 `Application startup complete` 后，按 `Ctrl-b d` detach。

## 服务验证

在 `search-agent-02` 上验证 embedding：

```bash
EMB_MODEL=/data/hf/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af

curl -sS --max-time 20 http://10.32.20.11:8004/v1/models

curl -sS --max-time 60 http://10.32.20.11:8004/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$EMB_MODEL\",\"input\":\"test query\"}" | head
```

在 `search-agent-02` 上验证 summary：

```bash
QWEN3_MODEL=/data/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

curl -sS --max-time 20 http://10.32.20.11:6010/v1/models

curl -sS --max-time 120 http://10.32.20.11:6010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$QWEN3_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Respond strictly as JSON with keys evidence and summary. Evidence: Paris is the capital of France.\"}],\"max_tokens\":128,\"temperature\":0,\"response_format\":{\"type\":\"json_object\"},\"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

## Ray 集群

在 `search-agent-02` 启动 Ray head：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate webagent-train

ray stop --force

ray start --head \
  --node-ip-address=10.32.56.200 \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --num-gpus=8
```

在 `search-agent-03` 加入 Ray：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate webagent-train

ray stop --force

ray start --address=10.32.56.200:6379 --num-gpus=8
```

回到 `search-agent-02` 检查：

```bash
ray status
```

期望状态：

- 2 个 active nodes
- `16.0 GPU`
- no pending nodes
- no failures

## 启动训练

在 `search-agent-02` 上用 tmux 启动：

```bash
tmux new -s train
```

在 tmux 里执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate webagent-train

REPO=/home/jovyan/code/searcherkit
cd "$REPO"
mkdir -p outputs/train_logs

set -o pipefail
TRIAL_NAME=qwen3_searcherkit_$(date +%Y%m%d_%H%M%S)
LOG=outputs/train_logs/${TRIAL_NAME}.log
TRIAL_NAME="$TRIAL_NAME" bash train/train_dist.sh 2>&1 | tee "$LOG"
```

训练开始后按 `Ctrl-b d` detach。重新进入：

```bash
tmux attach -t train
```

## Tmux 管理

查看所有 tmux session：

```bash
tmux ls
```

重新进入服务：

```bash
tmux attach -t embed
tmux attach -t summary
tmux attach -t train
```

正常 detach 是 `Ctrl-b d`。不要在服务进程里按 `Ctrl-C`，否则 server 会停止。
