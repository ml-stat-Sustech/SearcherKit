# Webagent

High-performance, widely compatible agent runtime.

## Key Notice

Implement with following focus
- Native concurrent
- good logging (global & per trace)
- error handling (recoverable / need to raise)
- response length (auto request model to generate final answer when reach content length)
- all support both hydra config and normal parameter passing (NO `args`)


## Agent Structure

The runtime is organized around an agent loop (`agent/`), model interfaces (`llm/`),
tool contracts (`tools/`), and configuration (`conf/`).

```text
src/webagent/
├── main.py                # Application entrypoint
├── pyproject.toml         # Project metadata and dependencies
├── conf/
│   └── config.yaml        # Hydra configs
├── agent/
│   └── react_agent.py     # Core ReAct-style agent orchestration
├── llm/
│   ├── client.py          # LLM API client adapter
│   ├── parser.py          # Model output parsing utilities
│   └── chat_types.py      # Chat/message data models
└── tools/
    └── base_tool.py       # Base tool interface and abstractions
```

## Module Responsibilities

- `agent/`: Coordinates planning, reasoning, and tool usage flow.
- `llm/`: Encapsulates model communication, response parsing, and message types.
- `tools/`: Defines tool contracts used by the agent runtime.
- `conf/`: Stores hydra configs template and default values.

## Config Imports

Config fields that are imported by the runtime use the `pkg://` scheme:
- Module only: `pkg://package.module`
- Module attribute: `pkg://package.module:ClassOrFunc`
