# Local Wiki Plugin

This plugin contains the active local wiki corpus path:

- `source.py`: reads MediaWiki XML or XML.bz2 dumps and normalizes pages into
  `IndexDocument` records with `title`, `text`, `url`, `links`, and metadata.
- `deploy_elasticsearch.py`: deploys normalized wiki documents to Elasticsearch,
  optionally with dense vectors in `text_vector`.

## Deploy To Elasticsearch

```bash
python -m searcherkit.plugins.local_wiki.deploy_elasticsearch \
  --wiki_dump_path /data/enwiki-pages-articles.xml.bz2 \
  --es_host http://127.0.0.1:9200 \
  --index_name wiki_qwen3 \
  --dense-vector \
  --model_name /models/Qwen3-Embedding-0.6B \
  --embedding_dim 1024 \
  --prompt_strategy qwen3 \
  --overwrite
```

The generated index uses the standard SearcherKit document fields and can be
queried through `searcherkit.sources.ElasticsearchSource`.
