"""Deploy a MediaWiki dump into an Elasticsearch dense-vector index."""

from __future__ import annotations

import argparse

from searchagent.plugins.indexing import deploy_to_elasticsearch
from searchagent.plugins.local_wiki.source import WikiDumpSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index a MediaWiki dump into Elasticsearch")
    parser.add_argument("--wiki_dump_path", required=True, help="Path to a MediaWiki XML or XML.bz2 dump")
    parser.add_argument("--es_host", required=True, help="Elasticsearch host URL")
    parser.add_argument("--index_name", required=True, help="Elasticsearch index name")
    parser.add_argument("--base_url", default="https://en.wikipedia.org/wiki/", help="Base URL for page links")
    parser.add_argument("--model_name", default="", help="SentenceTransformer model name or path")
    parser.add_argument("--embedding_dim", type=int, default=0, help="Dense vector dimension")
    parser.add_argument("--prompt_strategy", default="none", choices=("none", "e5", "qwen3"))
    parser.add_argument("--batch_size", "--cpu_batch_size", dest="batch_size", type=int, default=200)
    parser.add_argument("--embedding_batch_size", "--gpu_batch_size", dest="embedding_batch_size", type=int, default=16)
    parser.add_argument("--max_text_chars", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dense-vector", action="store_true", help="Enable dense vector indexing")
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    if cli.dense_vector and not cli.model_name:
        raise ValueError("--model_name is required with --dense-vector")
    if cli.dense_vector and cli.embedding_dim < 1:
        raise ValueError("--embedding_dim must be >= 1 with --dense-vector")

    source = WikiDumpSource(cli.wiki_dump_path, base_url=cli.base_url)
    indexed = deploy_to_elasticsearch(
        documents=source.iter_documents(limit=cli.limit),
        es_host=cli.es_host,
        index_name=cli.index_name,
        embedding_model_name=cli.model_name if cli.dense_vector else None,
        embedding_dim=cli.embedding_dim if cli.dense_vector else None,
        prompt_strategy=cli.prompt_strategy,
        overwrite=cli.overwrite,
        batch_size=cli.batch_size,
        embedding_batch_size=cli.embedding_batch_size,
        max_text_chars=cli.max_text_chars,
        shards=cli.shards,
        replicas=cli.replicas,
    )
    print(f"Indexed {indexed} wiki documents into {cli.index_name}")


if __name__ == "__main__":
    main()
