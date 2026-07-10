# SearchAgent

SearchAgent is a pluggable agent runtime for search-heavy tasks. It is designed
to work with different offline web/document sources, LLM providers, tool
servers, and evaluation recipes.

## Install

```bash
git clone https://github.com/ml-stat-Sustech//searchagent.git
cd searchagent
uv sync
```

Optional backend dependencies are split by plugin:

```bash
uv sync --extra elasticsearch-source
uv sync --extra vllm
uv sync --extra indexing
```

After installing the package, use the short `searchagent ...` command through
the console script. The module form works too:

```bash
python -m searchagent --help
```

## Quick Start

Before running, deploy the following services
- elasticsearch **< 9.0.0**
- embedding model
- LLM model

### Elasticsearch

Use docker:
```bash
local_path=/path/to/es/data
docker run \
  --name es-wiki \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -v ${local_path}:/usr/share/elasticsearch/data \
  -d elasticsearch:8.19.5

curl http://127.0.0.1:9200
```

Without docker:

- [Download](https://www.elastic.co/downloads/past-releases/elasticsearch-8-19-16)

- [Offical Doc](https://www.elastic.co/downloads/elasticsearch)

### Embedding Model

For deploying a local embedding model, we recommend using [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference), A blazing fast inference solution for text embeddings models.

### LLM Model

We recommend using [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang) to deploy your model. 

For better KV Cache hit rate, we recommand replace data parallel config with dedicated server setup. 

For example, for a `dp=2, tp=2` setup, start 2 seperate servers with `dp=1, tp=2`.

Pass the corresponding endpoint URLs in a list to the agent config:

```yaml
agent:
  llm_client:
    openai:
      base_url:
        - http://127.0.0.1:8001
        - http://127.0.0.1:8002
```

The agent runner will bind each session to a single endpoint to maximize cache hits.

### Deploy embedding database as search source
```bash
searchagent plugins deploy browsecomp-plus \
  --dataset_path Tevatron/browsecomp-plus-corpus \
  --es_host http://127.0.0.1:9200 \
  --index_name browsecomp_plus_qwen3 \
  --dense-vector \
  --model_name /models/Qwen3-Embedding-8B \
  --embedding_dim 4096 \
  --prompt_strategy qwen3 \
  --overwrite
```

### Start the agent with a bundled recipe:

```bash
searchagent run --config-path searchagent
```

## Usages

First verify that the CLI is installed and can see the top-level commands:

```bash
searchagent --help
```

Validate config fields against structured types before running:

```bash
searchagent inspect --config-path recipe/webexplorer --config-name webexplorer
searchagent inspect --config-path recipe/websailor --config-name websailor
```

The validator recursively compares every key in the composed config against the
corresponding dataclass fields (`RunConfig`, `SearchAgentConfig`, `ClientConfig`,
`SourceConfig`, `ToolConfig`, etc.) and reports:

- **ERROR** — unexpected field in config (not in target dataclass) or required field missing (no default value in target dataclass)

Hydra-style overrides are supported:

```bash
searchagent inspect --config-path recipe/webexplorer --config-name webexplorer agent.max_turn=50
```

Run the packaged default config:

```bash
searchagent run
```

This default config uses the packaged in-memory questions file at
`src/searchagent/config/memory_questions.jsonl` and writes outputs to
`outputs/agent_history`. It still needs a reachable LLM endpoint and API key.
Edit `src/searchagent/config/config.yaml` or pass Hydra-style overrides to point
the run at your own model endpoint, data file, or output directory:

```bash
searchagent run \
  agent.llm_client.model=Qwen3-8B \
  agent.llm_client.openai.base_url=http://127.0.0.1:8001/v1 \
  agent.llm_client.openai.api_key=EMPTY \
  output_path=outputs/demo
```

Run a benchmark recipe:

```bash
searchagent run --config-path recipe/webexplorer --config-name webexplorer
searchagent run --config-path recipe/websailor --config-name websailor
```

On Windows PowerShell, the same recipe commands can be written with backslashes:

```powershell
searchagent run --config-path recipe\webexplorer --config-name webexplorer
searchagent inspect --config-path recipe\websailor --config-name websailor
```

Evaluate saved run outputs with the LLM judge:

```bash
searchagent evaluate outputs/WebExplorer outputs/WebExplorer_eval --max-concurrency 32
```

List bundled plugins:

```bash
searchagent plugins list
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
- `searchagent.sources.build_source`: build a data source from a Hydra or
  plain Python config.
- `searchagent.tools.SearchTool` and `searchagent.tools.VisitTool`: expose a
  configured source to the agent as `search` and `visit`.
- `searchagent.llm`: add provider adapters for OpenAI-compatible servers, vLLM,
  or commercial APIs.
- `searchagent.plugins`: keep optional concrete backends out of the runtime
  core.

Native source-backed tools are wired by source name:

```yaml
agent:
  sources:
    - type: memory
      name: memory
      documents:
        - id: doc-1
          title: Example
          text: Example document body
  tools:
    - type: search
      name: search
      source:
        - memory
    - type: visit
      name: visit
      source:
        - memory
    - type: mcp
      name: web_search
      mcp_tool_name: search
      endpoint: http://127.0.0.1:8100/mcp
```

Elasticsearch-backed corpora use the same tool path:

```yaml
agent:
  sources:
    - type: elasticsearch
      name: bcp
      hosts: http://127.0.0.1:9200
      index: browsecomp_hybrid
      search_fields: [title, text]
      document_id_field: url
      fetch_field: url
      snippet_chars: 512
      metadata_fields: [links]
  tools:
    - type: search
      name: search
      source:
        - bcp
    - type: visit
      name: visit
      source:
        - bcp
```

See `docs/architecture.md` for the intended layering.

## Plugin Corpus Deployment

Plugins provide corpus readers, preprocessors, and Elasticsearch deployment
entry points for benchmark backends.

Wiki dump to Elasticsearch:

```bash
searchagent plugins deploy local-wiki \
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
searchagent plugins deploy browsecomp-plus \
  --dataset_path Tevatron/browsecomp-plus-corpus \
  --es_host http://127.0.0.1:9200 \
  --index_name browsecomp_plus_qwen3 \
  --dense-vector \
  --model_name /models/Qwen3-Embedding-8B \
  --embedding_dim 4096 \
  --prompt_strategy qwen3 \
  --overwrite
```

The direct module entry points are still available when you want to bypass the
top-level CLI dispatcher:

```bash
python -m searchagent.plugins.local_wiki.deploy_elasticsearch --help
python -m searchagent.plugins.browsecomp_plus.deploy_elasticsearch --help
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
