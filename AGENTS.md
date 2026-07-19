# AGENTS.md

## Additional Rules
- Do not use broad exception handlers such as `except Exception`.
  Catch explicit, concrete exception types instead.
- Do not add compatibility re-export layers by default. After moving code, update
  imports and config targets to the real module paths, then search for stale
  paths.
- After changing config, CLI, recipes, or plugin entry points, run at least a
  compose/help/import check. Do not stop at editing files.
- When running project commands, prefer an already existing/active virtual
  environment. Use `uv run` only when no usable environment can be found.
- For network-related commands or tests, use `uv run` even when a virtual
  environment exists. This includes real network calls, localhost services,
  proxies, and HTTP mocking/interception tools such as `respx` or mock
  transports.
- Always check `great-docs.yml` in any code changes for documentation configuration and consistency.

## Documentation Rules
- Use repository-relative links for links between project documentation files.
  Do not replace internal documentation links with GitHub, deployed-site, or
  machine-local absolute URLs.
- Prefix ordered documentation page filenames with a two-digit sequence and a
  hyphen, for example `01-search-files.qmd` and `02-search-web.qmd`. Keep the
  sequence consistent with the intended sidebar and reading order.
- Keep `index.qmd` as the unnumbered directory entry-point exception. Number
  the other pages in that directory when their relative order matters.
- After renaming a documentation page, update every relative link and search
  the repository for the stale filename before building the docs.

## Testing Rules
- New tests should cover three concerns when they apply: config initialization,
  functional behavior, and retry behavior.
- Config initialization means the same test flow should exercise construction
  through normal non-config arguments and through the config/factory path. For
  example, test the object once with direct constructor parameters and once with
  `config=...` or the module factory that consumes that config.
- Functional tests are the normal behavior checks for the object under test:
  inputs, outputs, request payloads, parsed responses, state changes, and error
  surfaces that are part of the object contract.
- Retry tests are additional functional tests for objects that can encounter
  recoverable errors such as timeouts, transient provider errors, or temporary
  source failures. When such errors exist, include both scenarios: the first
  `n-1` attempts fail and the `n`th attempt succeeds, and all `n` attempts fail
  and the final error is surfaced.

## Project Overview
SearcherKit is a pluggable search-agent runtime for retrieval-augmented tasks,
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
src/searcherkit/
|-- __main__.py              # thin `python -m searcherkit` entry, delegates to CLI
|-- __init__.py
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
|   |-- errors.py            # project-level exception taxonomy
|   |-- json_schema.py       # JSON Schema helpers for tool interfaces
|   |-- log.py               # logging, trace logging, log context
|   |-- messages.py          # provider-agnostic message structures
|   |-- retry.py             # retry config and wrappers
|   `-- utils.py
|-- config/
|   |-- config.yaml          # packaged default run config
|   |-- searcherkit.yaml     # example config
|   |-- agent/               # agent config groups
|   |-- common/              # retry/dataloader config groups
|   |-- examples/
|   |-- llm/                 # provider/parser config groups
|   |-- plugins/
|   |-- runtime/
|   |-- sources/
|   |-- tools/
|   `-- training/
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
|       |-- upstream.py      # provider-native tool-call parser
|       |-- websailor.py     # WebSailorParser
|       `-- webexplorer.py   # WebExplorerParser
|-- plugins/
|   |-- indexing.py          # shared Elasticsearch indexing helpers
|   |-- conversion/          # dataset/trace format conversion
|   |-- local_wiki/          # wiki source/preprocess/deploy
|   `-- browsecomp_plus/     # BrowseComp Plus source/preprocess/deploy
|-- runtime/
|   |-- batch.py             # async batch execution utilities
|   |-- runner.py            # AgentRunner and RunConfig
|   |-- evaluate.py          # LLM judge evaluation
|   |-- startup.py           # optional pre-run checks/startup
|   |-- checkpoint.py
|   |-- trace.py
|   `-- vllm_engine.py
|-- sources/
|   |-- base.py              # Source interface and SearchResult
|   |-- elasticsearch.py     # Elasticsearch-backed source
|   |-- factory.py           # source construction from config/plain params
|   |-- local_file.py        # local file source
|   |-- memory.py            # in-memory source for tests/simple runs
|   `-- web.py               # HTTP/web document source
|-- training/                # SFT/RL training integrations
`-- tools/
    |-- base.py              # Tool interface and tool-call structures
    |-- factory.py           # tool construction from config/plain params
    |-- mcp.py               # MCP-backed tool adapter
    |-- search.py            # source-backed search tool
    |-- summarizer.py        # optional tool-output summarization
    `-- visit.py             # source-backed visit tool

recipe/
|-- webexplorer/
|   `-- webexplorer.yaml
`-- websailor/
    `-- websailor.yaml
```

## Module Responsibilities
- `agent/`: agent reasoning, tool dispatch, context limits, final-answer control.
- Top-level `__main__.py` is intentionally kept at package root as the package
  entry point. Package-wide errors and observability live in `common/errors.py`
  and `common/log.py`.
- `cli/`: stable user-facing entry. It parses arguments, composes config, and
  calls existing runtime/plugin implementations. It must not duplicate indexing
  or benchmark logic.
- `common/`: cross-module utilities and data structures such as messages,
  config instantiation, retry, errors, logging, and dataloaders.
- `config/`: packaged defaults and reusable Hydra config groups. Do not put
  benchmark/paper-specific recipes here.
- `llm/`: LLM provider adapters. Add each provider as a dedicated module and
  register it in `llm/base.py:get_client`.
- `llm/parsers/`: model/training-format parsers. Put Qwen, WebSailor,
  WebExplorer, OpenAI tool-call, and similar parser implementations here.
- `plugins/`: data source reading, preprocessing, and Elasticsearch deployment.
  `searcherkit plugins deploy ...` should call these implementations rather than
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
python -m searcherkit --help
python -m searcherkit run --config-path webexplorer
python -m searcherkit inspect --config-path webexplorer
python -m searcherkit inspect --config-path websailor
python -m searcherkit evaluate outputs\webexplorer outputs\webexplorer_eval --max-concurrency 32
python -m searcherkit plugins list
python -m searcherkit plugins deploy local-wiki --help
python -m searcherkit plugins deploy browsecomp-plus --help
```

Hydra-style overrides are supported:
```powershell
python -m searcherkit inspect --config-path websailor agent.llm_client.model=demo
```

## Design Rules
- Keep provider adapters and parsers separate. Providers call APIs; parsers
  convert message and tool-call formats.
- Put run recipes in `recipe/`, plugin data/deploy logic in `plugins/`, and
  reusable defaults/config groups in `config/`.
- Do not put WebSailor/WebExplorer parser implementation code under `recipe/`;
  keep it in `src/searcherkit/llm/parsers/`.
- Do not restore `src/searcherkit/llm/client.py` or
  `src/searcherkit/llm/parser.py` compatibility re-export modules.
- Do not reintroduce `integrations/`; source ingestion and deployment belong in
  `src/searcherkit/plugins/`.
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
python -m compileall -q src\searcherkit recipe tests
python -m searcherkit --help
python -m searcherkit inspect --config-path webexplorer
python -m searcherkit inspect --config-path websailor
python -m pytest tests\cli\test_inspect.py tests\common\test_dataloader.py tests\tools\test_base.py tests\tools\test_multi_source_tools.py
python -m pytest tests\sources\test_memory.py tests\sources\test_local_file.py
python -m pytest tests\plugins\test_conversion.py tests\plugins\test_local_wiki.py tests\plugins\test_browsecomp_plus.py
uv run pytest tests\tools\test_search.py tests\tools\test_visit.py tests\sources\test_web.py
```

For tests that use real or mocked HTTP/network transports, use `uv run` even
when `.venv` exists, per the project command rule above. Examples include
`tests\cli\test_run.py`, `tests\tools\test_search.py`,
`tests\tools\test_visit.py`, `tests\sources\test_web.py`,
`tests\sources\test_elasticsearch.py`, `tests\llm\test_openai.py`, and
`tests\llm\test_anthropic.py`. Elasticsearch, Anthropic, and TUI dependencies
are included in the default installation; the `indexing` extra is reserved for
local indexing and deployment dependencies.

Search for stale paths:
```powershell
rg --pcre2 "WebAgent|searcherkit\.llm\.client\b|searcherkit\.llm\.parser(?!s)|integrations" src recipe tests docs README.md pyproject.toml
```

Pytest cache permission warnings may appear in this workspace; they are not
test failures by themselves.
