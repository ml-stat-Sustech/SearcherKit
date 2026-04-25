# SearchAgent Architecture

SearchAgent is organized as a small runtime core with pluggable data sources
and agent-callable tools.

```text
src/searchagent/
|-- agent/          # agent loops and orchestration
|-- common/         # messages, retry, config, dataloader utilities
|-- config/         # Hydra config templates
|-- plugins/        # concrete optional backends, such as local_wiki
|-- llm/            # LLM protocols, clients, and parsers
|-- runtime/        # batch runner, startup, evaluation, logging
|-- sources/        # data source contracts and adapters
`-- tools/          # search, visit, MCP, and domain tools
```

## Layering

`sources` define how SearchAgent connects to data. A source can be an offline
web index, vector database, local file corpus, SQL database, or a remote search
service. Each source implements `DataSource.search()` and `DataSource.fetch()`.

`tools` define what the agent can call. The native `SearchTool` and `VisitTool`
wrap one configured source and expose it as the model-facing `search` and
`visit` actions. MCP remains available as a transport/integration tool, but it
is not the core abstraction for search.

```text
Agent
  -> tools.SearchTool / tools.VisitTool
  -> sources.DataSource
  -> Elasticsearch / FAISS / Chroma / local wiki / custom corpus
```

The built-in `ElasticsearchSource` handles common web/document indexes with
`title`, `text`, `url`, and optional metadata fields. BrowseComp Plus indexes
created by `plugins/browsecomp_plus/deploy_elasticsearch.py` are wired by
configuring `document_id_field` and `fetch_field` to `url`.

## Configuration Flow

1. Configure one or more named `agent.sources` entries.
2. Configure `agent.tools` entries that reference a source name.
3. `SearchAgent` builds sources first, then passes them to the tool factory.
4. The agent calls `search` or `visit`; the tool delegates to its configured
   source.

For tools, `type` selects the implementation and `name` is the model-visible
function name. This keeps MCP-backed tools from colliding with built-in
`search` and `visit` implementations.

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

BrowseComp Plus through Elasticsearch:

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

## Plugin Deployment

`searchagent.plugins.local_wiki` reads MediaWiki XML/XML.bz2 dumps, extracts
plain text plus internal links, and can deploy the normalized documents to
Elasticsearch with optional dense vectors.

`searchagent.plugins.browsecomp_plus` reads Hugging Face datasets or local
JSON/JSONL/parquet files, normalizes common title/text/url/id fields, and uses
the same Elasticsearch vector indexing helper.

Both plugin deployment CLIs write `title`, `text`, `url`, `links`, optional
`metadata`, and optional `text_vector`, so they can be queried through the
built-in `ElasticsearchSource`.

## Extension Flow

1. Implement `searchagent.sources.DataSource` for a backend.
2. Configure `SearchTool` and `VisitTool` with that source.
3. Add optional backend-specific code under `plugins/` or an external
   package.
4. Keep agent logic independent from the concrete data source.
