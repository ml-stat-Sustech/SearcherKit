# SearchAgent

SearchAgent is a pluggable agent runtime for search-heavy tasks. It is designed
to work with different offline web/document sources, LLM providers, tool
servers, and evaluation recipes.

## Install

```bash
git clone https://github.com/your-org/searchagent.git
cd searchagent
uv sync
uv pip install -e .
```

Optional backend dependencies are split by plugin:

```bash
uv sync --extra elasticsearch
uv sync --extra vllm
uv sync --extra local-wiki
```

## Quick Start

Edit `src/searchagent/config/config.yaml` or create a new Hydra config:

```bash
python -m searchagent --config-name=config
```

## Project Layout

```text
src/searchagent/
|-- agent/          # agent loops and orchestration
|-- common/         # messages, retry, config, dataloader utilities
|-- config/         # Hydra config templates
|-- plugins/        # optional concrete backends, such as local_wiki
|-- llm/            # LLM protocols, clients, and parsers
|-- runtime/        # batch runner, startup, evaluation, logging
|-- sources/        # data source contracts and adapters
`-- tools/          # agent-callable search, visit, and MCP tools
```

## Extension Points

- `searchagent.sources.DataSource`: implement this to support a new data source
  such as Elasticsearch, FAISS, Chroma, SQLite, local wiki, or a custom corpus.
- `searchagent.sources.build_sources`: build a name-to-source map from Hydra or
  plain Python configs.
- `searchagent.tools.SearchTool` and `searchagent.tools.VisitTool`: expose a
  configured source to the agent as `search` and `visit`.
- `searchagent.llm`: add provider adapters for OpenAI-compatible servers, vLLM,
  Ollama, local Transformers, or commercial APIs.
- `searchagent.plugins`: keep optional concrete backends out of the runtime
  core.

Native source-backed tools are wired by source name:

```yaml
agent:
  sources:
    - name: memory
      target: pkg://searchagent.sources:MemorySource
      documents:
        - id: doc-1
          title: Example
          text: Example document body
  tools:
    - type: search
      name: search
      source: memory
    - type: visit
      name: visit
      source: memory
    - type: mcp
      name: web_search
      mcp_tool_name: search
      endpoint: http://127.0.0.1:8100/mcp
```

Elasticsearch-backed corpora use the same tool path:

```yaml
agent:
  sources:
    - name: bcp
      target: pkg://searchagent.sources:ElasticsearchSource
      hosts: http://127.0.0.1:9200
      index: browsecomp_hybrid
      search_fields: [title^2, text]
      document_id_field: url
      fetch_field: url
      highlight_fragment_size: 128
      snippet_chars: 512
      metadata_fields: [links]
  tools:
    - type: search
      name: search
      source: bcp
    - type: visit
      name: visit
      source: bcp
```

See `docs/architecture.md` for the intended layering.

## Plugin Corpus Deployment

Plugins provide corpus readers, preprocessors, and Elasticsearch deployment
entry points for benchmark backends.

Wiki dump to Elasticsearch:

```bash
python -m searchagent.plugins.local_wiki.deploy_elasticsearch \
  --wiki_dump_path /data/enwiki-pages-articles.xml.bz2 \
  --es_host http://127.0.0.1:9200 \
  --index_name wiki_qwen3 \
  --dense-vector \
  --model_name /models/Qwen3-Embedding-0.6B \
  --embedding_dim 1024 \
  --prompt_strategy qwen3 \
  --overwrite
```

BrowseComp Plus to Elasticsearch:

```bash
python -m searchagent.plugins.browsecomp_plus.deploy_elasticsearch \
  --dataset_path Tevatron/browsecomp-plus-corpus \
  --es_host http://127.0.0.1:9200 \
  --index_name browsecomp_plus_qwen3 \
  --dense-vector \
  --model_name /models/Qwen3-Embedding-8B \
  --embedding_dim 4096 \
  --prompt_strategy qwen3 \
  --overwrite
```

## Logging

Per-run logging defaults to:

- `output_path/run.log` for global run-level logs
- `output_path/traces/*.log` for per-sample trace logs

Trace behavior is configured in YAML:

```yaml
logging:
  global_file: ${output_path}/run.log
  trace:
    enabled: true
    dir: ${output_path}/traces
    format: text
    level: DEBUG
    filename_template: "{sample_id}_{trace_id}.log"
```

Set `logging.trace.format: json` to emit structured JSON traces.
