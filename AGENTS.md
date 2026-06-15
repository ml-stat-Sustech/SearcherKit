# AGENTS.md

## Additional Rules
- Do not use broad exception handlers such as `except Exception`.
  Catch explicit, concrete exception types instead.
- Do not add compatibility re-export layers by default. After moving code, update
  imports and config targets to the real module paths, then search for stale
  paths.
- Do not add overly defensive programming. If input/output behavior is unclear,
  first inspect the code statically or run a small test to confirm whether that
  behavior is actually possible before adding exception handling.
- After changing config, CLI, recipes, or plugin entry points, run at least a
  compose/help/import check. Do not stop at editing files.

## Project Overview
SearchAgent is a pluggable search-agent runtime for retrieval-augmented tasks,
benchmark recipes, source plugins, Elasticsearch deployment, and multiple LLM
provider adapters.

Implementation priorities:
- Native concurrency and stable batch execution.
- Complete logging, including global logs and per-trace logs.
- Explicit error handling for recoverable, fatal, provider, source, and tool
  failures.
- Context/turn limit handling that asks the model to produce a final answer when
  limits are reached.
- Support both Hydra config and normal parameter passing. Avoid adding opaque
  `args` objects to internal APIs.
- Keep recipe, plugin, runtime, provider, and parser responsibilities separate.

## Current Layout
```text
src/searchagent/
|-- __main__.py              # thin `python -m searchagent` entry, delegates to CLI
|-- __init__.py
|-- errors.py                # project-level exception taxonomy
|-- log.py                   # logging, trace logging, log context
|-- agent/
|   |-- base.py              # agent protocol/base types
|   |-- search_agent.py      # SearchAgent and SearchAgentConfig
|   |-- react_agent.py       # ReAct-style example agent
|   `-- single_turn_agent.py # single-turn agent for judge/evaluate usage
|-- cli/
|   |-- main.py              # CLI dispatcher
|   |-- run.py               # run config/recipe
|   |-- evaluate.py          # evaluate saved outputs
|   |-- plugins.py           # plugin discovery/deploy entry
|   |-- inspect.py           # config validation
|   `-- config.py            # Hydra compose and ConfigStore registration
|-- common/
|   |-- config.py            # import/instantiate helpers
|   |-- dataloader.py        # generic dataloader
|   |-- messages.py          # provider-agnostic message structures
|   `-- retry.py             # retry config and wrappers
|-- config/
|   |-- config.yaml          # packaged default run config
|   |-- searchagent.yaml     # example config
|   |-- agent/               # agent config groups
|   |-- common/              # retry/dataloader config groups
|   |-- llm/                 # provider/parser config groups
|   |-- runtime/
|   |-- sources/
|   `-- tools/
|-- llm/
|   |-- base.py              # Client/ClientConfig/get_client/provider configs
|   |-- openai.py            # OpenAI-compatible client
|   |-- dashscope.py         # DashScope adapter
|   |-- vllm.py              # vLLM adapter
|   |-- ollama.py            # Ollama adapter
|   |-- anthropic.py         # Anthropic placeholder adapter
|   |-- transformers.py      # local Transformers placeholder adapter
|   `-- parsers/
|       |-- base.py          # Parser/ParserConfig/ParsingError/get_parser
|       |-- qwen.py          # QwenParser
|       |-- websailor.py     # WebSailorParser
|       `-- webexplorer.py   # WebExplorerParser
|-- plugins/
|   |-- indexing.py          # shared Elasticsearch indexing helpers
|   |-- local_wiki/          # wiki source/preprocess/deploy
|   `-- browsecomp_plus/     # BrowseComp Plus source/preprocess/deploy
|-- runtime/
|   |-- agent_runner.py      # lower-level single agent execution helper
|   |-- batch.py             # async batch execution utilities
|   |-- runner.py            # AgentRunner and RunConfig
|   |-- evaluate.py          # LLM judge evaluation
|   |-- errors.py            # runtime-specific exceptions
|   |-- startup.py           # optional pre-run checks/startup
|   |-- checkpoint.py
|   |-- trace.py
|   `-- vllm_engine.py
|-- sources/
|   |-- base.py              # Source interface and SearchResult
|   |-- elasticsearch.py     # Elasticsearch-backed source
|   |-- factory.py           # source construction from config/plain params
|   `-- memory.py            # in-memory source for tests/simple runs
`-- tools/
    |-- base.py              # Tool interface and tool-call structures
    |-- factory.py           # tool construction from config/plain params
    |-- mcp.py               # MCP-backed tool adapter
    `-- search.py            # source-backed search/visit tools

recipe/
|-- webexplorer/
|   `-- webexplorer.yaml
`-- websailor/
    `-- websailor.yaml
```

## Module Responsibilities
- `agent/`: agent reasoning, tool dispatch, context limits, final-answer control.
- Top-level `__main__.py`, `errors.py`, and `log.py` are intentionally kept at
  package root because they are package-wide entry, error, and observability
  surfaces. Do not bury them in a feature subpackage unless the whole public
  boundary changes.
- `cli/`: stable user-facing entry. It parses arguments, composes config, and
  calls existing runtime/plugin implementations. It must not duplicate indexing
  or benchmark logic.
- `common/`: cross-module utilities and data structures such as messages,
  config instantiation, retry, and dataloaders.
- `config/`: packaged defaults and reusable Hydra config groups. Do not put
  benchmark/paper-specific recipes here.
- `llm/`: LLM provider adapters. Add each provider as a dedicated module and
  register it in `llm/base.py:get_client`.
- `llm/parsers/`: model/training-format parsers. Put Qwen, WebSailor,
  WebExplorer, OpenAI tool-call, and similar parser implementations here.
- `plugins/`: data source reading, preprocessing, and Elasticsearch deployment.
  `searchagent plugins deploy ...` should call these implementations rather than
  reimplementing them.
- `recipe/`: benchmark, paper, or experiment-level run recipes. Recipes may
  reference parsers, plugins, sources, and tools, but should not contain core
  implementation code.
- `runtime/`: batch execution, checkpointing, trace serialization, evaluation,
  and optional startup checks.
- `sources/`: searchable data source abstractions and implementations.
- `tools/`: callable tool abstractions, MCP tools, and source-backed search/visit
  tools.

## CLI
Common commands:
```powershell
python -m searchagent --help
python -m searchagent run --config-path recipe\webexplorer --config-name webexplorer
python -m searchagent inspect --config-path recipe\webexplorer --config-name webexplorer
python -m searchagent inspect --config-path recipe\websailor --config-name websailor
python -m searchagent evaluate outputs\webexplorer outputs\webexplorer_eval --max-concurrency 32
python -m searchagent plugins list
python -m searchagent plugins deploy local-wiki --help
python -m searchagent plugins deploy browsecomp-plus --help
```

Hydra-style overrides are supported:
```powershell
python -m searchagent inspect --config-path recipe\websailor --config-name websailor agent.llm_client.model=demo
```

## Design Rules
- Keep provider adapters and parsers separate. Providers call APIs; parsers
  convert message and tool-call formats.
- Put run recipes in `recipe/`, plugin data/deploy logic in `plugins/`, and
  reusable defaults/config groups in `config/`.
- Do not put WebSailor/WebExplorer parser implementation code under `recipe/`;
  keep it in `src/searchagent/llm/parsers/`.
- Do not restore `src/searchagent/llm/client.py` or
  `src/searchagent/llm/parser.py` compatibility re-export modules.
- Do not reintroduce `integrations/`; source ingestion and deployment belong in
  `src/searchagent/plugins/`.
- Do not reintroduce local wiki `mcp/` or `retrievers/` directories unless there
  is a new, actively used integration path. Current wiki/BrowseComp Plus flows
  should go through sources, tools, and Elasticsearch deployment helpers.
- When deleting or moving modules, search for stale imports and stale
  `pkg://...` config targets.
- This project is commonly edited on Windows/PowerShell; be careful with paths
  and quoting.

## Suggested Verification
Use targeted checks:
```powershell
python -m compileall -q src\searchagent recipe tests\test_config_instantiation.py
python -m searchagent --help
python -m searchagent inspect --config-path recipe\webexplorer --config-name webexplorer
python -m searchagent inspect --config-path recipe\websailor --config-name websailor
python -m pytest tests\test_config_instantiation.py tests\test_source_tools.py tests\test_plugins_sources.py tests\test_elasticsearch_source.py
```

Search for stale paths:
```powershell
rg --pcre2 "webagent|WebAgent|searchagent\.llm\.client\b|searchagent\.llm\.parser(?!s)|integrations" src recipe tests docs README.md pyproject.toml
```

Pytest cache permission warnings may appear in this workspace; they are not
test failures by themselves.
