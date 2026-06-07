# AReaL 训练参数迁移到 slime 的对照

本文对照旧 AReaL 训练配置与当前 slime 启动脚本，判断哪些参数已经迁移、哪些刚补齐、哪些仍然是框架差异或需要单独确认。

相关文件：

- AReaL shell: `train/train_dist_areal.sh`
- AReaL config: `train/train_dist_areal.yaml`
- slime shell: `train/train_dist_slime.sh`
- slime agent config: `train/searchagent_slime_agent.yaml`

## 总体结论

核心训练参数已经迁移到 slime：

- 模型路径
- 训练数据路径
- 2 nodes x 8 GPUs
- rollout batch size 128
- 每 prompt 采样 8 条
- 训练 6 epoch
- rollout response length 8192
- context length 65536
- temperature/top_p
- learning rate 5e-6
- GRPO
- save interval 32
- SearchAgent prompt/parser/tool/source 配置
- 训练期间 evaluation 的 valid dataset、eval interval、eval agent/source

本次对照后又补齐了几项之前漏掉或不够等价的参数：

- `seed=1`
- `rollout_seed=1`
- `train_dataset.shuffle=true` -> `--rollout-shuffle`
- `ppo_n_minibatches=4` -> `--num-steps-per-rollout 4`
- 对应 `global_batch_size` 从 `128` 调整为 `256`
- `eps_clip_higher=0.28` -> `--eps-clip-high 0.28`
- `adv_norm` -> `--normalize-advantages`
- `clip_grad=1.0`、`eps_clip=0.2`、`kl=0.0` 显式写入 slime shell
- `valid_dataset` / `eval_workflow` -> `--eval-prompt-data` + `eval_agent`
- `evaluator.freq_steps=32` -> `--eval-interval 32`

仍然不是 100% 等价，主要差异来自框架自身：

- AReaL 的 FSDP backend 字符串、weight update mode、Megatron/SGLang 细节，在 slime 中没有直接同名同义参数。
- AReaL 的一些 actor/ref microbatch token 约束没有直接映射到 slime，需要通过 Megatron/slime 的 batch/token 参数另配。

## Shell override 对照

| AReaL 参数 | AReaL 值 | slime 参数 | slime 值 | 状态 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `cluster.n_nodes` | `2` | `--actor-num-nodes` | `2` | 已映射 | actor 训练节点数 |
| `cluster.n_gpus_per_node` | `8` | `--actor-num-gpus-per-node` | `8` | 已映射 | 每节点训练 GPU |
| `rollout.backend` | `sglang:d8` | `--rollout-num-gpus` / `--rollout-num-gpus-per-engine` | `8` / `8` | 近似映射 | slime 通过 rollout GPU 数和每 engine GPU 数表达 |
| `actor.path` | `/home/jovyan1/Qwen3-8B` | `HF_CHECKPOINT`, `--hf-checkpoint`, `--ref-load` | `/home/jovyan1/Qwen3-8B` | 已映射 | actor/ref/model/tokenizer 基本同源 |
| `train_dataset.path` | `/home/jovyan1/...jsonl` | `PROMPT_DATA`, `--prompt-data` | 同路径 | 已映射 | 训练 prompt 数据 |
| `train_dataset.batch_size` | `128` | `--rollout-batch-size` | `128` | 已映射 | AReaL 的 prompt batch 对应 slime rollout prompt batch |
| `gconfig.n_samples` | `8` | `--n-samples-per-prompt` | `8` | 已映射 | 每个 prompt 采样数 |
| `total_train_epochs` | `6` | `--num-epoch` | `6` | 已映射 | slime 会据此计算 rollout step |
| `seed` | `1` | `--seed` | `1` | 已补齐 | 训练随机种子 |
| `sglang.random_seed` | `${seed}` | `--rollout-seed` | `1` | 已补齐 | rollout 采样/数据 shuffle seed |
| `train_dataset.shuffle` | `true` | `--rollout-shuffle` | enabled | 已补齐 | slime 默认不 shuffle，需要显式打开 |
| `workflow.agent.llm_client.default_kwargs.max_completion_tokens` | `8192` | `--rollout-max-response-len` | `8192` | 已映射 | rollout 最大生成长度 |
| `workflow.agent.max_tokens` / `sglang.context_length` | `65536` | `--rollout-max-context-len` | `65536` | 已映射 | rollout context window |
| `temperature` | `1.0` | `--rollout-temperature` | `1.0` | 已映射 | 采样温度 |
| `top_p` | `1.0` | `--rollout-top-p` | `1.0` | 已映射 | nucleus sampling |
| `actor.optimizer.lr` | `5e-6` | `--lr` | `5e-6` | 已映射 | 学习率 |
| `actor.optimizer.gradient_clipping` | `1.0` | `--clip-grad` | `1.0` | 已补齐 | 梯度裁剪 |
| `actor.eps_clip` | `0.2` | `--eps-clip` | `0.2` | 已补齐/显式化 | slime 默认也是 0.2，但现在显式写出 |
| `actor.eps_clip_higher` | `0.28` | `--eps-clip-high` | `0.28` | 已补齐 | 之前没有传，slime 会默认等于 eps_clip |
| `actor.kl_ctl` | `0.0` | `--kl-coef` | `0.0` | 已补齐/显式化 | slime 默认也是 0.0，但现在显式写出 |
| `actor.ppo_n_minibatches` | `4` | `--num-steps-per-rollout` | `4` | 已补齐 | slime 注释说明可用它控制每个 rollout 分几个训练 step |
| rollout 总样本数 | `128 * 8 = 1024` | `--global-batch-size` | `256` | 已修正 | `1024 / 4 = 256`，对应 4 个 minibatch/steps |
| `saver.freq_steps` | `32` | `--save-interval` | `32` | 已映射 | 保存间隔 |
| `evaluator.freq_steps` | `32` | `--eval-interval` | `32` | 已补齐 | 训练期间 eval 间隔 |
| `valid_dataset.path` | `/home/jovyan1/browsecomp_plus_decrypted_qa.jsonl` | `VALID_DATA`, `--eval-prompt-data browsecomp_plus ...` | 同路径 | 已补齐 | slime 用 eval dataset config，不直接传 Dataset 对象 |
| `eval_gconfig.n_samples` | `1` | `--n-samples-per-eval-prompt` | `1` | 已补齐 | eval 每 prompt 采样数 |
| `eval_gconfig.temperature/top_p` | `1.0 / 1.0` | `--eval-temperature` / `--eval-top-p` | `1.0 / 1.0` | 已补齐 | eval 采样参数 |
| `eval_gconfig.max_new_tokens` | `8192` | `--eval-max-response-len` | `8192` | 已补齐 | eval 最大生成长度 |
| `eval_workflow.agent.max_tokens` | `65536` | `--eval-max-context-len` + `eval_agent.max_tokens` | `65536` | 已补齐 | eval context |
| `dynamic_filter_fn` | `searchagent.training.rewards.should_accept` | `--dynamic-sampling-filter-path` | `searchagent.training.slime_rollout.mixed_reward_filter` | 近似映射 | 都是保留 mixed reward group；slime 版本对连续 reward 更通用 |

## Agent 环境参数对照

这些参数已经从 `train_dist_areal.yaml` 迁到 `searchagent_slime_agent.yaml`：

| AReaL agent 配置 | slime agent 配置 | 状态 |
| --- | --- | --- |
| `llm_client.model=${actor.path}` | `llm_client.model=/home/jovyan1/Qwen3-8B` | 已映射 |
| `temperature=1.0` | `temperature=1.0` | 已映射 |
| `max_completion_tokens=8192` | `max_completion_tokens=8192` | 已映射 |
| `top_p=1.0` | `top_p=1.0` | 已映射 |
| `enable_thinking=true` | `enable_thinking=true` | 已映射 |
| `parser.type=qwen` | `parser.type=qwen` | 已映射 |
| `qwen.upstream_parsed=true` | `qwen.upstream_parsed=true` | 已映射 |
| `qwen.drop_thinking=false` | `qwen.drop_thinking=false` | 已映射 |
| `max_tokens=65536` | `max_tokens=65536` | 已映射 |
| `system_prompt` | `system_prompt` | 已映射 |
| `query_prompt` | `query_prompt` | 已映射 |
| `max_tokens_prompt_margin=11000` | `max_tokens_prompt_margin=11000` | 已映射 |
| `max_turn=50` | `max_turn=50` | 已映射 |
| `raise_repeat_tool_call=false` | `raise_repeat_tool_call=false` | 已映射 |
| Elasticsearch source | Elasticsearch source | 已映射 |
| search tool schema | search tool schema | 已映射 |

一个小差异：

- AReaL training workflow 的 `tool_retry_config.exceptions` 只有 `RecoverableError`。
- 当前 slime agent YAML 里包含 `RecoverableError` 和 `jsonschema.ValidationError`。

这不会改变正常路径，但在 tool 参数校验错误时，slime 训练侧可能会重试更多错误类型。如果要严格贴齐 AReaL training 行为，可以把 slime training agent 的 `jsonschema.ValidationError` 去掉，并另建一个 eval agent config。

## 训练期间 Evaluation 对照

AReaL 旧配置里训练期间 evaluation 是完整开启的：

- `valid_dataset.path=/home/jovyan1/browsecomp_plus_decrypted_qa.jsonl`
- `eval_workflow.agent.max_turn=1000`
- `eval_workflow.agent.sources[0].index=browsecomp_plus_qwen3-embedding-8b`
- `eval_gconfig.n_samples=1`
- `evaluator.freq_steps=32`

slime 侧现在对应补为：

```bash
VALID_DATA="${VALID_DATA:-/home/jovyan1/browsecomp_plus_decrypted_qa.jsonl}"

--eval-prompt-data browsecomp_plus "${VALID_DATA}"
--eval-input-key question
--eval-label-key answer
--n-samples-per-eval-prompt 1
--eval-max-response-len 8192
--eval-max-context-len 65536
--eval-temperature 1.0
--eval-top-p 1.0
--eval-interval 32
--searchagent-eval-agent-config-key eval_agent
```

`train/searchagent_slime_agent.yaml` 中新增了 `eval_agent`，用来对齐 AReaL 的 `eval_workflow`：

- source index: `browsecomp_plus_qwen3-embedding-8b`
- max turn: `1000`
- max tokens: `65536`
- same prompt/parser/generation kwargs

这里的数据读取方式和 AReaL 不一样：

- AReaL 在 `train_dist_areal.py` 中显式 `load_dataset` 并把 `valid_dataset` 传给 `PPOTrainer`。
- slime 在参数解析阶段把 `--eval-prompt-data` 解析为 `EvalDatasetConfig`，然后在 `slime.rollout.sglang_rollout.eval_rollout_single_dataset` 里构造 `Dataset`。

## Reward / normalization 对照

| AReaL 参数 | AReaL 值 | slime 状态 | 说明 |
| --- | --- | --- | --- |
| `reward=f1` | F1 | 已映射 | `slime_rollout.py` 中 `_score_history` 使用同样 F1 + overlong penalty |
| `overlong_penalty_margin=5000` | workflow 字段 | 间接使用 | 当前 reward 用 `agent.max_tokens_prompt_margin / 2`，与旧 workflow 一致 |
| `reward_norm.mean_level=group` | group mean | slime 默认 | slime GRPO 默认 `rewards_normalization=True`，按 prompt group 减均值 |
| `reward_norm.std_level=group` | group std | slime 默认 | slime GRPO 默认 `grpo_std_normalization=True`，按 prompt group 除 std |
| `reward_norm.group_size=${gconfig.n_samples}` | 8 | 已映射 | `n_samples_per_prompt=8` |
| `adv_norm.mean_level=batch/std_level=batch` | batch advantage norm | 已补齐 | `--normalize-advantages` |

## 尚未完整迁移或需要确认的参数

这些参数没有直接一一映射，或者需要依赖 slime/Megatron/SGLang 的独立配置：

| AReaL 参数 | 当前 slime 状态 | 原因 / 建议 |
| --- | --- | --- |
| `actor.backend=fsdp:d4c2` | 未直接映射 | slime 使用 Megatron backend，FSDP backend 字符串没有同名同义参数 |
| `actor.weight_update_mode=xccl` | 未直接映射 | slime 有 `--update-weight-mode`，但语义不是简单等价；需要按 slime 权重同步模式确认 |
| `actor.attn_impl=flash_attention_3` | 未直接映射 | slime/Megatron/SGLang attention backend 需要按模型和容器环境确认 |
| `actor.disable_dropout=true` | 未显式映射 | 可能由 Megatron/slime 默认或模型 config 控制，需确认 |
| `actor.gradient_checkpointing=true` | 未显式映射 | slime/Megatron 有自己的 checkpointing 参数体系，需单独确认 |
| `actor.dtype=bfloat16` / `ref.dtype=bfloat16` / `sglang.dtype=${actor.dtype}` | 未显式映射 | slime/Megatron 通常有 bf16/fp16 参数；当前没显式传 |
| `actor.mb_spec.max_tokens_per_mb=90000` | 未映射 | slime 对应更像 `--max-tokens-per-gpu` / dynamic batch；当前用固定 `--micro-batch-size 1` |
| `ref.mb_spec.max_tokens_per_mb=70000` | 未映射 | ref/logprob 侧 token budget 需要通过 slime/Megatron 参数确认 |
| `actor.recompute_logprob=true` | 未显式映射 | slime 训练端会按需要计算 current logprob；rollout old logprob 由 Sample 提供。语义不完全等价 |
| `actor.use_decoupled_loss=true` | 未映射 | slime loss 实现不同，没有直接同名参数 |
| `actor.rejection_sampling.upper=5.0` | 未映射 | 当前只做 mixed reward dynamic filter，没有 AReaL rejection_sampling 上界语义 |
| `rollout.max_concurrent_rollouts=256` | 未直接映射 | slime 并发由 `sglang_server_concurrency`、rollout batch、engine 数等控制；当前未传 256 |
| `rollout.request_timeout=7200` | 未完全映射 | slime router 默认 timeout 是 14400；如需严格等价可考虑 `--sglang-router-request-timeout-secs 7200` |
| `sglang.mem_fraction_static=0.8` | 未显式映射 | 可能对应 prefixed SGLang server arg，但需在实际 slime/SGLang 环境确认参数名 |
| `sglang.max_prefill_tokens=65536` | 未显式映射 | 可能对应 prefixed SGLang server arg，但需验证 |
| `stats_logger.wandb` | 未迁移 | 当前 slime shell 没有显式 `--wandb-project` 等 |
| `perf_tracer.enabled=false` | 未迁移 | slime tracing/logging 参数不同 |

## 目前 slime shell 中的关键参数

当前 `train/train_dist_slime.sh` 的核心参数为：

```bash
--hf-checkpoint /home/jovyan1/Qwen3-8B
--ref-load /home/jovyan1/Qwen3-8B
--prompt-data /home/jovyan1/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean.jsonl
--eval-prompt-data browsecomp_plus /home/jovyan1/browsecomp_plus_decrypted_qa.jsonl
--eval-interval 32
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--rollout-num-gpus 8
--rollout-num-gpus-per-engine 8
--rollout-batch-size 128
--n-samples-per-prompt 8
--num-epoch 6
--seed 1
--rollout-seed 1
--rollout-shuffle
--global-batch-size 256
--num-steps-per-rollout 4
--micro-batch-size 1
--rollout-max-response-len 8192
--rollout-max-context-len 65536
--rollout-temperature 1.0
--rollout-top-p 1.0
--lr 5e-6
--clip-grad 1.0
--eps-clip 0.2
--eps-clip-high 0.28
--kl-coef 0.0
--normalize-advantages
--save-interval 32
--advantage-estimator grpo
```

## 判断

如果只看“主要实验超参”和“agent 环境行为”，现在已经基本迁移到 slime。

如果要求“训练系统内部行为完全复刻 AReaL”，还没有完全等价，尤其是：

- Megatron/FSDP/backend 并行策略
- weight update 模式
- dtype/attention/checkpointing/token microbatch
- SGLang engine 细节
- wandb/stats/perf tracing

这些需要在真实 slime Docker/runtime 环境里根据 slime 支持的参数继续校准，不能仅靠文件名一一替换。
