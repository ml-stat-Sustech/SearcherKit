# AReaL 与 slime 在 SearchAgent Agentic RL 训练中的实现对照

本文按 agentic RL 的训练流程，对比当前代码中 AReaL 和 slime 如何实现同一个目标。
核心结论是：

- **agent 环境逻辑可以共用**：prompt、parser、tool/source、multi-turn loop、context/turn limit、final answer、reward 语义。
- **trajectory 打包层需要按框架分别适配**：AReaL 需要返回 interaction/export 格式；slime 需要填充 `Sample.tokens`、`loss_mask`、`rollout_log_probs`、`reward` 等字段。
- 因此当前代码保持了两个框架的训练胶水层分离，但共用同一个 `SearchAgentTraining` agent runtime。

## 入口命名

| 用途 | AReaL | slime | 说明 |
| --- | --- | --- | --- |
| Shell 入口 | `train/train_dist_areal.sh` | `train/train_dist_slime.sh` | 两边显式区分，方便对照 |
| Python 入口 | `src/searchagent/training/train_dist_areal.py` | `src/searchagent/training/train_dist_slime.py` -> `train_slime.py` | AReaL 保留旧 PPOTrainer 路径；slime 使用 slime 的 Ray/rollout/training model 路径 |
| 主配置 | `train/train_dist_areal.yaml` | shell CLI 参数 + `train/searchagent_slime_agent.yaml` | AReaL 是完整 YAML；slime 主要通过 CLI 驱动训练，agent 配置单独放 YAML |
| 默认兼容入口 | `train/train_dist.sh` | 转发到 slime | 默认脚本现在走 slime；要跑旧 AReaL 用 `train_dist_areal.sh` |
| 训练期间评估 | `valid_dataset` + `eval_workflow` + `evaluator.freq_steps` | `--eval-prompt-data` + `eval_agent` + `--eval-interval` | 两边都在训练期间评估；slime 需要显式配置 eval dataset 和 eval agent |

## 总体数据流

### AReaL 路径

```text
train_dist_areal.sh
  -> python -m searchagent.training.train_dist_areal
  -> load_expr_config(..., SearchAgentARealTrainingConfig)
  -> datasets.load_dataset(train/valid)
  -> PPOTrainer.train(...)
  -> ARealSearchAgentWorkflow.arun_episode(engine, data)
  -> ArealOpenAI + ARealClient
  -> SearchAgentTraining.run(question)
  -> tool/search multi-turn interaction
  -> reward 计算
  -> areal_client.set_last_reward(...)
  -> areal_client.export_interactions(...)
  -> PPOTrainer 消费 interaction/token/logprob/reward
```

### slime 路径

```text
train_dist_slime.sh
  -> python -m searchagent.training.train_dist_slime
  -> train_slime.main / slime.utils.arguments.parse_args
  -> create_placement_groups / create_rollout_manager / create_training_models
  -> eval dataset config from --eval-prompt-data
  -> rollout_manager.generate(...)
  -> slime custom generate hook: generate_searchagent(...)
  -> SlimeSGLangClient
  -> SearchAgentTraining.run(question)
  -> tool/search multi-turn interaction
  -> reward 计算
  -> merge_turns(...) / fan_out_sample_segments(...)
  -> 填充 slime Sample
  -> Data Buffer / actor_model.async_train(...) 消费 Sample
```

## 按 agentic RL 训练流程逐步对照

| 流程阶段 | 目标 | AReaL 实现 | slime 实现 | 共用/差异 |
| --- | --- | --- | --- | --- |
| 1. 训练启动 | 选择训练框架、资源、模型、数据路径 | `train/train_dist_areal.sh` 调 `python -m searchagent.training.train_dist_areal --config train_dist_areal.yaml ...` | `train/train_dist_slime.sh` 调 `python -m searchagent.training.train_dist_slime ...` 并传 slime CLI 参数 | 入口分离，避免一个脚本同时承载两套框架语义 |
| 2. 训练配置解析 | 得到 trainer、rollout、actor、dataset、workflow 参数 | `train_dist_areal.py` 调 AReaL 的 `load_expr_config`，动态构造继承 `GRPOConfig` 的 SearchAgent 配置类 | `train_slime.py` 调 slime 的 `parse_args(add_searchagent_slime_arguments)`，SearchAgent 额外参数通过 `slime_args.py` 注册 | AReaL 是 Hydra/YAML 风格；slime 是 CLI args + agent YAML |
| 3. prompt 数据读取 | 从数据集中拿 `question` 和 `answer` | `train_dist_areal.py` 里显式 `load_dataset("json", data_files=...)`，再传给 `PPOTrainer` | slime 框架按 `--prompt-data`、`--input-key question`、`--label-key answer` 构造 DataSource 和 `Sample` | 数据语义一致，但读取责任在不同地方 |
| 4. 创建训练主循环 | 管 rollout、训练、保存、评估 | AReaL 的 `PPOTrainer` 负责训练循环 | `train_slime.py` 创建 placement groups、rollout manager、actor/critic model，并在 loop 中调用 `generate` 和 `async_train` | 训练调度完全不同 |
| 5. 进入 rollout episode | 对一个 question 跑一次 agent 环境 | AReaL 调 `ARealSearchAgentWorkflow.arun_episode(engine, data)` | slime 调 custom generate hook `generate_searchagent(args, sample, sampling_params, evaluation)` | 都是框架回调用户代码 |
| 6. 构造 LLM client | 把框架推理引擎适配成 SearchAgent 的 `Client` | `ArealOpenAI(engine=..., tokenizer=...)` 再包成 `ARealClient` | `SlimeSGLangClient` 调 slime/SGLang router `/generate` | 这是框架适配层，不是 agent 环境 |
| 7. 运行 agent 环境 | 执行 prompt、parser、tools、search、multi-turn loop | `SearchAgentTraining(config=agent_config, llm_client=client)` 后 `agent.run(data["question"])` | `SearchAgentTraining(config=agent_config, llm_client=client)` 后 `agent.run(_question_from_sample(sample), session_id=...)` | 这是核心共用部分 |
| 8. tool/search 交互 | 让模型多轮调用 search tool，拼接 tool response | 由 `SearchAgentTraining` / `SearchAgent` 完成 | 同样由 `SearchAgentTraining` / `SearchAgent` 完成 | 完全同一套环境逻辑 |
| 9. 异常与 episode 状态 | 识别格式错误、重复查询、tool call 过多、context 超限 | `workflow.py` 捕获 `RepeatedToolCallError`、`TooManyToolCallsError`、`LLMOutputError`、`ParsingError`、`LLMContextError`，写 AReaL stats | `slime_rollout.py` 捕获同类错误，写入 metadata/status | 错误语义相同，状态承载不同 |
| 10. token/logprob 采集 | 得到训练需要的 token 和 rollout logprob | AReaL 由 `ArealOpenAI/ARealClient` 内部记录 interaction，最后 `export_interactions` 输出 | `SlimeSGLangClient._generate` 显式传 `return_logprob=True`，从 `output_token_logprobs` 取 token id 和 logprob，保存为 `TurnRecord` | 这是最大差异之一 |
| 11. 多轮 trajectory 拼接 | 把多轮 assistant generation 和中间 tool response 变成一条训练序列 | AReaL 使用自己的 interaction export 格式，workflow 只需要返回 `areal_client.export_interactions(style=...)` | slime 使用 `merge_turns(client.turns, metadata=...)`，把多轮 prompt/output 拼成 `TokenSegment`；tool response 等非模型生成内容 loss mask 为 0 | slime 侧显式做 loss mask 对齐 |
| 12. reward 计算 | 根据最终答案和长度惩罚给 episode reward | `workflow.py` 从最后 assistant message 解析 `\boxed{...}`，算 F1 和 overlong penalty | `slime_rollout.py` 的 `_score_history` 做同样事情 | reward 逻辑基本可共用 |
| 13. reward 写回 | 让训练框架能读到 reward | `areal_client.set_last_reward(final_reward)`，再 `apply_reward_discount` | `fan_out_sample_segments(..., reward, ...)` 写入 `sample.reward`；`custom_rm` 作为 fallback | reward 的载体不同 |
| 14. loss mask | 控制哪些 token 参与 policy loss | AReaL interaction exporter 内部负责按 interaction 格式表达 | slime 明确写 `Sample.loss_mask`；模型输出 token 为 1，环境/tool/context tail 为 0 | slime 侧更显式 |
| 15. rollout 数据进入训练 | 把 rollout 产物送入 PPO/GRPO 更新 | AReaL trainer 消费 workflow 返回的 `InteractionWithTokenLogpReward` | slime rollout manager 把 `Sample` 放入 rollout 数据路径/Data Buffer，然后 actor `async_train` 消费 | 训练数据格式不同 |
| 16. dynamic filtering | 过滤没有训练信号的 sample group | AReaL 在 `trainer.train(..., dynamic_filter_fn=config.dynamic_filter_fn)` 中传入过滤函数 | slime 用 `--dynamic-sampling-filter-path searchagent.training.slime_rollout.mixed_reward_filter` | 同一目的，不同 hook |
| 17. evaluation | 评估时跑同样 agent 逻辑，但通常样本数/配置不同 | AReaL 用 `eval_workflow`，仍然是 `ARealSearchAgentWorkflow` | slime 支持 evaluation 参数和 eval dataset config；`generate_searchagent(..., evaluation=True)` 会选择 eval agent config key | agent 逻辑共用，入口不同 |
| 18. logging/stats | 记录 turn、context、format、reward 等指标 | AReaL 用 `stats_tracker` 和 `workflow_context.stat_scope()` | slime 主要放进 `Sample.metadata`，同时 slime 框架记录 rollout/train metrics | 统计承载不同 |

### 训练期间 Evaluation 的数据读取差异

AReaL 的 train/eval 数据都是在 SearchAgent 自己的训练入口里显式读取：

```python
train_dataset = load_dataset("json", data_files=config.train_dataset.path, split="train")
valid_dataset = load_dataset("json", data_files=config.valid_dataset.path, split="train")
PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset)
```

slime 的训练集读取不在 SearchAgent 入口里，而是在 slime 的 rollout data source 中完成：

```text
RolloutManager
  -> data_source_cls = load_function(args.data_source_path)
  -> RolloutDataSourceWithBuffer(args)
  -> RolloutDataSource.__init__
  -> Dataset(args.prompt_data, prompt_key=args.input_key, label_key=args.label_key, ...)
  -> get_samples(...)
```

slime 的 eval 数据也类似，不是传 `valid_dataset` 对象，而是通过 CLI 转成 `EvalDatasetConfig`：

```text
--eval-prompt-data browsecomp_plus /home/jovyan1/browsecomp_plus_decrypted_qa.jsonl
--eval-input-key question
--eval-label-key answer
--n-samples-per-eval-prompt 1
--eval-interval 32
```

训练时 `train_slime.py` 会在 rollout loop 中周期性调用：

```python
if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
    ray.get(rollout_manager.eval.remote(rollout_id))
```

当前 slime 侧还配置了 `--searchagent-eval-agent-config-key eval_agent`。这样 eval rollout 使用
`train/searchagent_slime_agent.yaml` 中的 `eval_agent`，对应 AReaL 的 `eval_workflow`：

- eval 数据：`/home/jovyan1/browsecomp_plus_decrypted_qa.jsonl`
- eval source index：`browsecomp_plus_qwen3-embedding-8b`
- eval samples：`1`
- eval max turn：`1000`
- eval interval：`32`

## 关键代码角色

### 共用的 agent 环境代码

这些代码是 AReaL 和 slime 都应该复用的环境逻辑：

- `src/searchagent/training/agent.py`
  - `SearchAgentTraining`
  - 限制单轮最多一个 tool call
  - 可选检测重复 tool call
- `src/searchagent/agent/search_agent.py`
  - multi-turn agent loop
  - prompt 构造
  - parser 调用
  - tool dispatch
  - context/turn limit 处理
- `src/searchagent/tools/`
  - search/visit tool 抽象和实现
- `src/searchagent/sources/`
  - Elasticsearch/source 抽象
- `src/searchagent/llm/parsers/`
  - Qwen/WebSailor/WebExplorer 等 parser
- `src/searchagent/training/rewards.py`
  - F1、overlong penalty、dynamic filter 等 reward 相关逻辑

这部分不应该因为换训练框架而大改。换框架时主要换的是 LLM client 适配器和 rollout exporter。

### AReaL 专用胶水层

- `src/searchagent/training/train_dist_areal.py`
  - 创建 AReaL 配置类
  - 调 `load_expr_config`
  - `load_dataset`
  - 创建 `PPOTrainer`
  - 把 `ARealSearchAgentWorkflow` 传给 trainer
- `src/searchagent/training/workflow.py`
  - 定义 `ARealSearchAgentWorkflow`
  - `arun_episode` 中创建 `ArealOpenAI`
  - 跑 `SearchAgentTraining`
  - 计算 reward
  - 调 `areal_client.export_interactions`
- `src/searchagent/training/areal_client.py`
  - 把 AReaL 的 OpenAI-compatible client 包装成 SearchAgent 的 `Client`

AReaL 的特点是：trajectory 的 token/logprob/reward 对齐更多依赖 `ArealOpenAI` 的 interaction 记录和 `export_interactions`。

### slime 专用胶水层

- `src/searchagent/training/train_dist_slime.py`
  - slime 命名入口，转到 `train_slime.main`
- `src/searchagent/training/train_slime.py`
  - 解析 slime CLI
  - 创建 Ray placement groups
  - 创建 rollout manager 和 training models
  - 调 rollout generate 和 actor train
- `src/searchagent/training/slime_args.py`
  - 给 slime CLI 注册 SearchAgent 专用参数
  - 如 `--searchagent-agent-config`
- `src/searchagent/training/slime_client.py`
  - 把 slime/SGLang router 包装成 SearchAgent 的 `Client`
  - 调 `/generate`
  - 请求 `return_logprob=True`
  - 把每轮生成保存为 `TurnRecord`
- `src/searchagent/training/slime_rollout.py`
  - `generate_searchagent` 是 slime custom generate hook
  - 从 slime `Sample` 取 question/label
  - 跑 `SearchAgentTraining`
  - 算 reward
  - `merge_turns` / `fan_out_sample_segments`
  - 填 `Sample.tokens`、`response_length`、`loss_mask`、`rollout_log_probs`、`reward`、`status`

slime 的特点是：rollout 数据最终必须落到 slime 的 `Sample` 结构上，由 Data Buffer / actor training path 消费。

## token / loss mask / reward 对齐细节

### AReaL

AReaL workflow 中主要代码路径是：

```python
areal_client = ArealOpenAI(...)
client = ARealClient(client=areal_client, ...)
agent = SearchAgentTraining(config=agent_config, llm_client=client)
await agent.run(data["question"])
...
areal_client.set_last_reward(final_reward)
areal_client.apply_reward_discount(self.reward_discount)
return areal_client.export_interactions(style=self.export_style)
```

这里 workflow 负责：

- 调 agent
- 解析最终答案
- 计算 reward
- 把 reward 写入 AReaL client
- 返回 AReaL 需要的 interaction export

token ids、logprob、interaction turn 的细粒度记录主要由 AReaL 的 OpenAI-compatible client 维护。

### slime

slime rollout 中主要代码路径是：

```python
client = SlimeSGLangClient(...)
agent = SearchAgentTraining(config=agent_config, llm_client=client)
history = await agent.run(...)
reward, metadata = _score_history(...)
segment = merge_turns(client.turns, metadata=metadata)
out = fan_out_sample_segments(sample, [segment], reward, state.tokenizer, metadata=metadata)
```

`SlimeSGLangClient` 每次 LLM complete 时：

```python
output = await post(
    self.url,
    {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    },
)
output_ids = [item[1] for item in output_token_logprobs]
output_log_probs = [float(item[0]) for item in output_token_logprobs]
return TurnRecord(prompt_ids=..., output_ids=..., output_log_probs=...)
```

然后 `merge_turns` 做多轮拼接：

- 第一个 turn 的 prompt 作为 `Sample` prompt tokens
- 后续 turn 的 prompt 中，相对前面新增的 tool response / user context 作为 response tail 追加，但 `loss_mask=0`
- 每个 assistant generation 的 output token 追加到 response，`loss_mask=1`
- rollout logprob 只对模型生成 token 有效；非模型 token 置 0

最后写入 slime `Sample`：

- `sample.tokens = prompt_ids + response_ids`
- `sample.response_length = len(response_ids)`
- `sample.loss_mask = loss_mask`
- `sample.rollout_log_probs = rollout_log_probs`
- `sample.response = tokenizer.decode(response_ids, ...)`
- `sample.reward = reward`
- `sample.status = COMPLETED/TRUNCATED/FAILED`

## 为什么不能只写一个完全共用 workflow

因为两个框架的训练输入格式不同：

- AReaL 训练器要的是 AReaL interaction export，里面包含它自己需要的 token/logprob/reward/版本等信息。
- slime 训练器要的是 slime `Sample`，并且 Data Buffer 和 actor train path 读取的是 `Sample.tokens`、`loss_mask`、`rollout_log_probs`、`reward` 等字段。

如果强行共用一个 exporter，会出现两个问题：

1. exporter 里会充满框架条件分支，反而更难读。
2. 一旦某个框架升级字段格式，另一个框架的路径也容易被误伤。

更稳妥的结构是：

```text
共用：
  SearchAgentTraining / SearchAgent / tools / sources / parser / reward helpers

AReaL 适配：
  ARealClient + ARealSearchAgentWorkflow + export_interactions

slime 适配：
  SlimeSGLangClient + generate_searchagent + Sample 打包
```

## 当前代码中两边行为对齐的关键点

为了让 AReaL 和 slime 的 rollout 环境尽量一致，需要关注：

1. **prompt 一致**
   - AReaL: `train/train_dist_areal.yaml` 的 `workflow.agent.system_prompt/query_prompt`
   - slime: `train/searchagent_slime_agent.yaml` 的 `agent.system_prompt/query_prompt`

2. **parser 一致**
   - 两边都应使用 Qwen parser，且 `upstream_parsed/drop_thinking` 设置一致。

3. **tool/source 一致**
   - Elasticsearch host、index、embedding model、summary model、search tool schema 要一致。

4. **generation kwargs 一致**
   - temperature、top_p、max_completion_tokens、enable_thinking 要一致。

5. **answer format 一致**
   - 当前 reward 解析 `\boxed{...}`。
   - 如果 prompt 改成 `<answer>...</answer>`，AReaL/slime 两边 reward parser 都要一起改。

6. **max context / max turn 一致**
   - AReaL 用 config 中的 `max_tokens/max_turn/max_tokens_prompt_margin`。
   - slime agent config 也要保持同样值；另外 slime CLI 里 `--rollout-max-context-len` 也要匹配。

7. **reward filter 一致**
   - AReaL 通过 `dynamic_filter_fn`。
   - slime 通过 `mixed_reward_filter`。

8. **evaluation 一致**
   - AReaL: `valid_dataset` + `eval_workflow` + `evaluator.freq_steps=32`。
   - slime: `--eval-prompt-data` + `eval_agent` + `--eval-interval 32`。

## 当前差异与风险点

1. **slime 侧 trajectory 打包是新适配层**
   - `merge_turns` 的逻辑需要在真实 rollout 上验证。
   - 特别是多轮 tool response 后 prompt suffix 的拼接和 loss mask 是否完全符合训练预期。

2. **AReaL 侧保留旧 workflow**
   - `train_dist_areal.yaml` 与原 `train_dist.yaml` 内容一致。
   - `train_dist_areal.sh` 的启动参数已对齐旧脚本。

3. **默认入口已变化**
   - `train/train_dist.sh` 现在转发到 slime。
   - 跑 AReaL 必须显式用 `train/train_dist_areal.sh`。

4. **环境依赖不同**
   - AReaL 需要 `areal`、`datasets` 等。
   - slime 需要 `sglang_router`、`sglang`、Ray、Megatron/slime runtime 等。
   - 当前本地环境缺这些完整训练依赖，所以只能做 compile/help/shell 检查。

5. **eval 路径需要在真实 slime runtime 下验证**
   - 当前已经补了 eval dataset 和 eval agent config。
   - 但本地缺 `sglang_router`，还没有跑完整 slime CLI parse 和真实训练中 eval。

6. **slime Docker 与本地 `./slime` 版本需要对齐**
   - 当前代码参考的是本地 `./slime` 0.3.0 API。
   - 如果 Docker 中 slime 版本不同，优先确认 `Sample`、`GenerateState`、`custom_generate_function_path`、`TurnRecord/merge_turns` 相关 API 是否一致。

## 可以进一步抽象的部分

如果后续想减少重复，可以考虑抽出一个框架无关的 episode runner：

```python
async def run_searchagent_episode(agent_config, llm_client, question, label):
    agent = SearchAgentTraining(config=agent_config, llm_client=llm_client)
    history = await agent.run(question)
    reward, metadata = score_history(...)
    return EpisodeResult(history=history, reward=reward, metadata=metadata, agent_stats=...)
```

然后：

- AReaL adapter 把 `EpisodeResult` 写到 `ArealOpenAI` interaction export。
- slime adapter 把 `EpisodeResult` 和 `SlimeSGLangClient.turns` 写到 `Sample`。

但不建议把 AReaL exporter 和 slime Sample packer 合成一个函数，因为两边训练格式本质不同。

## 一句话版本

AReaL 和 slime 在当前代码里实现的是同一个 agentic RL 目标：

```text
question -> SearchAgent 多轮工具交互 -> final answer -> reward -> policy update
```

但它们落地方式不同：

- AReaL 的关键是 `workflow -> export_interactions`。
- slime 的关键是 `custom generate -> Sample tokens/loss_mask/logprobs/reward`。

所以 agent 环境本身应当共用；trajectory 序列化和训练框架接口应当分别写适配器。
