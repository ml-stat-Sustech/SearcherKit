# SearchAgent CLI

SearchAgent exposes a thin command-line interface for running recipes, evaluating
saved outputs, and invoking bundled plugin utilities.

## Run A Recipe

Run the packaged default config:

```bash
searchagent run
```

Run a benchmark recipe:

```bash
searchagent run --config-path recipe/webexplorer --config-name webexplorer
searchagent run --config-path recipe/websailor --config-name websailor
```

Pass Hydra-style overrides after the command:

```bash
searchagent run --config-path recipe/webexplorer --config-name webexplorer \
  agent.llm_client.openai.base_url=http://127.0.0.1:8001/v1 \
  dataloader.source=data/browsecomp_plus_decrypted_qa.jsonl \
  output_path=outputs/webexplorer
```

## Inspect Config

Print the final composed config without running anything:

```bash
searchagent inspect config --config-path recipe/webexplorer --config-name webexplorer
```

Overrides work here too:

```bash
searchagent inspect config --config-path recipe/websailor --config-name websailor \
  agent.llm_client.model=WebSailor-7B
```

## Evaluate Outputs

Evaluate saved run records with the LLM judge:

```bash
searchagent evaluate outputs/webexplorer outputs/webexplorer_eval --max-concurrency 32
```

The judge uses `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

## Plugins

List bundled plugins:

```bash
searchagent plugins list
```

Deploy a local wiki dump to Elasticsearch:

```bash
searchagent plugins deploy local-wiki \
  --wiki_dump_path data/wiki.xml.bz2 \
  --es_host http://127.0.0.1:9200 \
  --index_name wiki_local \
  --overwrite
```

Deploy BrowseComp Plus to Elasticsearch:

```bash
searchagent plugins deploy browsecomp-plus \
  --dataset_path data/browsecomp_plus_corpus.jsonl \
  --es_host http://127.0.0.1:9200 \
  --index_name browsecomp_plus \
  --overwrite
```

Enable dense-vector indexing by adding:

```bash
--dense-vector --model_name Qwen3-Embedding-0.6B --embedding_dim 1024
```
