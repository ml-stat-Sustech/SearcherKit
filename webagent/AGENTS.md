# AGENTS.md

## Additional Rules
- 禁止使用泛型异常捕获（如 `except Exception`）。需要捕获时必须使用明确、具体的异常类型。

## 项目概览（来自 README.md + 目录探索）
Webagent 是高性能、兼容性强的 agent runtime。实现时需关注：
- 原生并发
- 完善日志（全局与每个 trace）
- 错误处理（可恢复 / 需要抛出）
- 响应长度控制（达到内容长度时自动请求模型生成最终回答）
- 同时支持 Hydra 配置与普通参数传递（禁止 `args`）

## 目录结构与入口
```text
src/webagent/
├── __main__.py            # 包入口
├── __init__.py            # 包初始化
├── log.py                 # 日志配置与工具
├── conf/
│   └── config.yaml        # Hydra 配置模板
├── agent/
│   ├── agent.py           # Agent 抽象/接口
│   ├── react_agent.py     # ReAct 风格编排
│   └── webagent.py        # WebAgent 实现
├── llm/
│   ├── client.py          # LLM 客户端适配
│   ├── parser.py          # 输出解析
│   └── chat_types.py      # Chat/Message 数据模型
├── tools/
│   └── tool.py            # Tool 接口定义
├── runtime/
│   ├── agent_runner.py    # 运行时调度
│   └── vllm_engine.py     # vLLM 引擎适配
├── data_source/
│   ├── source.py          # 数据源基类/接口
│   └── generic.py         # 通用数据源实现
└── utils/
    ├── async_utils.py     # 异步工具
    ├── config.py          # 配置工具
    └── retry.py           # 重试工具
```

## 模块职责速览
- `agent/`: 规划、推理、工具调用的主流程协调。
- `llm/`: 模型通信、解析与消息类型。
- `tools/`: 工具契约与接口。
- `conf/`: Hydra 配置模板与默认值。
- `runtime/`: 运行时调度与引擎适配。
- `data_source/`: 数据源抽象与实现。
- `utils/`: 通用工具函数（异步、配置、重试）。
