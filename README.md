# SearchAgent

> Build, evaluate, and train search agents without locking your research to a
> single model, corpus, tool protocol, or training framework.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd.svg)](https://hydra.cc/)
[![License](https://img.shields.io/badge/license-see%20repository-lightgrey.svg)](#license)

SearchAgent is a modular runtime for search-intensive agents. It brings agent
rollouts, heterogeneous search sources, tool execution, model adapters,
benchmark recipes, evaluation, and RL training integration into one coherent
stack.

Use it to reproduce deep-search systems, compare models on the same retrieval
environment, connect private corpora, or generate trajectories for SFT and RL.
The runtime stays independent of any single provider or training framework, so
experiments remain reusable as the ecosystem changes.

## Why SearchAgent?

- **Training-ready agent runtime** — integrate search rollouts with AReaL and
  Slime while retaining control over LLM calls, exceptions, retries, and final
  answer generation.
- **One tool protocol, many backends** — run the same agent against memory,
  local files, web pages, Elasticsearch, MCP servers, or custom sources.
- **Model and parser independence** — keep provider API calls separate from
  Qwen, WebSailor, WebExplorer, and provider-native tool-call formats.
- **Research-grade batch execution** — native concurrency, stable session-to-
  endpoint affinity, checkpointing, global logs, and per-sample traces.
- **Reproducible experiments** — compose agents, models, sources, tools, and
  benchmark recipes with Hydra, then validate the final config before launch.
- **Designed to be extended** — add a source, tool, provider, parser, plugin,
  or recipe without rewriting the agent loop.

## Training integrations

SearchAgent is built to serve as the rollout and tool-use layer around modern
post-training systems.

| Algorithm | AReaL result | Slime result |
| --- | :---: | :---: |
| GRPO | `<RESULT>` | `<RESULT>` |
| IGPO | `<RESULT>` | `<RESULT>` |

The integration surface is intentionally explicit: training systems can hook
LLM generation while SearchAgent owns message state, tool dispatch, recoverable
errors, retry policy, turn limits, and context-limit finalization.

Explore the training guides:

- [RL with AReaL](user-guide/training/rl-areal.qmd)
- [RL with Slime](user-guide/training/rl-slime.qmd)
- [Supervised fine-tuning](user-guide/training/sft.qmd)

LLM client hook and exception-control example:

```python
# <CODE EXAMPLE>
```

## From question to grounded answer

```text
Question
   │
   ▼
Agent loop ─────► LLM client ─────► Parser
   ▲                                  │
   │                                  ▼
   └──── Tool result ◄──── Search / Visit / MCP
                              │
                              ▼
                  File · Web · Elasticsearch · Custom
```

Providers call model APIs. Parsers normalize model-specific output. Sources
own retrieval. Tools expose source capabilities to the agent. This separation
makes every layer independently replaceable and testable.

## Quick start

SearchAgent requires Python 3.12 or newer.

```bash
git clone https://github.com/ml-stat-Sustech/searchagent.git
cd searchagent
uv sync
```

Inspect a bundled research recipe before running it:

```bash
uv run searchagent inspect \
  --config-path recipe/webexplorer \
  --config-name webexplorer
```

Run the agent against your model endpoint:

```bash
uv run searchagent run \
  --config-path recipe/webexplorer \
  --config-name webexplorer \
  agent.llm_client.model=Qwen3-8B \
  agent.llm_client.openai.base_url=http://127.0.0.1:8001/v1 \
  agent.llm_client.openai.api_key=EMPTY
```

The complete setup guide covers Elasticsearch, embedding services, multiple
LLM endpoints, evaluation, logging, and Windows/PowerShell commands:
[SearchAgent Guide](user-guide/index.qmd).

## Bring your own search environment

Sources and tools are wired by name, so an agent recipe can move between local
documents and production indexes without changing the agent implementation.

```yaml
agent:
  sources:
    - type: elasticsearch
      name: knowledge_base
      hosts: http://127.0.0.1:9200
      index: documents
      search_fields: [title, text]
      document_id_field: url

  tools:
    - type: search
      name: search
      source: [knowledge_base]
    - type: visit
      name: visit
      source: [knowledge_base]
```

Built-in plugin workflows can prepare and deploy benchmark corpora:

```bash
uv run searchagent plugins list
uv run searchagent plugins deploy local-wiki --help
uv run searchagent plugins deploy browsecomp-plus --help
```

See [Source and Tool](user-guide/inference/source-tool.qmd) for the
runtime model and extension points.

## Model and benchmark results

SearchAgent evaluates both general-purpose LLMs and models trained specifically
for deep research across web-search and wiki-QA environments.

| Category | Benchmarks / models |
| --- | --- |
| Web search | BCP, BrowseComp, BrowseComp zh |
| Wiki QA | GAIA, HotpotQA, DeepSearchQA |
| General models | Qwen, Gemma, GPT, DeepSeek, Claude |
| Search models | Tongyi-DeepResearch, OpenResearcher, Openseeker, DR-Venus, WebExplorer, SlimSearcher |

| Model | BCP | BrowseComp | BrowseComp zh | GAIA | HotpotQA | DeepSearchQA |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `<MODEL>` | `<RESULT>` | `<RESULT>` | `<RESULT>` | `<RESULT>` | `<RESULT>` | `<RESULT>` |
| `<MODEL>`¹ | `<RESULT>` | `<RESULT>` | `<RESULT>` | `<RESULT>` | `<RESULT>` | `<RESULT>` |

¹ Reported by the original paper. Unmarked rows are SearchAgent reproductions.
Each reproduction is accompanied by its runnable code and configuration.

<details>
<summary>Model compatibility code example</summary>

```python
# <CODE EXAMPLE>
```

</details>

<details>
<summary>Model compatibility config example</summary>

```yaml
# <CONFIG EXAMPLE>
```

</details>

## Architecture at a glance

```text
src/searchagent/
├── agent/       agent loops, tool dispatch, and final-answer control
├── llm/         provider clients and model/training-format parsers
├── sources/     memory, file, web, Elasticsearch, and custom retrieval
├── tools/       search, visit, multi-source, summarizer, and MCP tools
├── plugins/     corpus conversion, preprocessing, and deployment
├── runtime/     batch execution, checkpoints, evaluation, and tracing
├── config/      reusable Hydra config groups
└── training/    SFT/RL integration points
```

Benchmark- and paper-specific configurations live in `recipe/`, keeping the
core runtime reusable across experiments.

## Examples and demos

| Topic | Code | Config | Demo |
| --- | --- | --- | --- |
| RL training | `<CODE_LINK>` | `<CONFIG_LINK>` | — |
| Model and tool compatibility | `<CODE_LINK>` | `<CONFIG_LINK>` | — |
| Source and plugin development | `<CODE_LINK>` | `<CONFIG_LINK>` | `<VIDEO_LINK>` |
| Secondary development | `<CODE_LINK>` | `<CONFIG_LINK>` | — |
| TUI | `<CODE_LINK>` | `<CONFIG_LINK>` | `<VIDEO_LINK>` |

## Documentation

- [Full guide](user-guide/index.qmd)
- [Parser and LLM adapters](user-guide/inference/parser-llm.qmd)
- [Source and tool system](user-guide/inference/source-tool.qmd)
- [Search execution flow](user-guide/inference/start-to-search.qmd)
- [CLI reference](docs/cli/index.qmd)

## Contributing

SearchAgent is actively evolving. Reproduction reports, new model parsers,
source adapters, benchmark recipes, training integrations, bug reports, and
documentation improvements are welcome. Open an issue with the model,
benchmark, and environment you want to support—or submit a focused pull
request.

If SearchAgent helps your research or makes your agent stack easier to reason
about, consider starring the repository. It helps more search-agent builders
find the project.

## License

See the repository license file for terms.
