"""Shared Elasticsearch indexing helpers for corpus plugins."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IndexDocument:
    """Normalized document shape used by plugin preprocessors."""

    id: str
    title: str
    text: str
    url: str
    links: list[Mapping[str, Any]] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_source(self) -> dict[str, Any]:
        source = {
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "links": list(self.links),
        }
        if self.metadata:
            source["metadata"] = dict(self.metadata)
        return source


def apply_embedding_prompt(text: str, strategy: str = "none") -> str:
    if strategy == "none":
        return text
    if strategy == "e5":
        return f"passage: {text}"
    if strategy == "qwen3":
        return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nPassage:{text}"
    raise ValueError(f"unknown prompt strategy: {strategy!r}")


def iter_batches(items: Iterable[IndexDocument], batch_size: int) -> Iterator[list[IndexDocument]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[IndexDocument] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_elasticsearch_client(hosts: str, *, request_timeout: float = 100.0) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise ImportError(
            "Elasticsearch deployment requires the 'elasticsearch' package. "
            "Install with `uv sync --extra elasticsearch-source` or `uv sync --extra indexing`."
        ) from exc
    return Elasticsearch(hosts, request_timeout=request_timeout)


def create_elasticsearch_index(
    client: Any,
    *,
    index_name: str,
    embedding_dim: int | None = None,
    vector_field: str = "text_vector",
    shards: int = 1,
    replicas: int = 0,
    overwrite: bool = False,
) -> None:
    if not index_name:
        raise ValueError("index_name must be non-empty")
    if embedding_dim is not None and embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")

    if client.indices.exists(index=index_name):
        if not overwrite:
            return
        client.indices.delete(index=index_name)

    mappings: dict[str, Any] = {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "text": {"type": "text", "analyzer": "standard"},
            "url": {"type": "keyword"},
            "links": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text", "index": False},
                    "target": {"type": "keyword", "index": False},
                    "url": {"type": "keyword", "index": False},
                },
            },
            "metadata": {"type": "object", "enabled": True},
        }
    }
    if embedding_dim is not None:
        mappings["properties"][vector_field] = {
            "type": "dense_vector",
            "dims": embedding_dim,
            "index": True,
            "similarity": "cosine",
        }

    client.indices.create(
        index=index_name,
        settings={"index": {"number_of_shards": shards, "number_of_replicas": replicas}},
        mappings=mappings,
    )


def load_sentence_transformer(model_name: str) -> Any:
    if not model_name:
        raise ValueError("model_name must be non-empty when vector indexing is enabled")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Vector indexing requires the 'sentence-transformers' package. "
            "Install with `uv sync --extra indexing`."
        ) from exc
    return SentenceTransformer(model_name, trust_remote_code=True)


def encode_documents(
    model: Any,
    documents: list[IndexDocument],
    *,
    prompt_strategy: str,
    max_text_chars: int,
    batch_size: int,
) -> list[list[float]]:
    if max_text_chars < 1:
        raise ValueError("max_text_chars must be >= 1")
    texts = [
        apply_embedding_prompt(document.text[:max_text_chars], prompt_strategy)
        for document in documents
    ]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def index_documents(
    client: Any,
    *,
    index_name: str,
    documents: Iterable[IndexDocument],
    embedding_model: Any | None = None,
    vector_field: str = "text_vector",
    prompt_strategy: str = "none",
    batch_size: int = 200,
    embedding_batch_size: int = 16,
    max_text_chars: int = 32768,
) -> int:
    try:
        from elasticsearch.helpers import bulk
    except ImportError as exc:
        raise ImportError(
            "Bulk indexing requires the 'elasticsearch' package. "
            "Install with `uv sync --extra elasticsearch-source` or `uv sync --extra indexing`."
        ) from exc

    total = 0
    for batch in iter_batches(documents, batch_size):
        vectors = None
        if embedding_model is not None:
            vectors = encode_documents(
                embedding_model,
                batch,
                prompt_strategy=prompt_strategy,
                max_text_chars=max_text_chars,
                batch_size=embedding_batch_size,
            )

        actions = []
        for idx, document in enumerate(batch):
            source = document.to_source()
            if vectors is not None:
                source[vector_field] = vectors[idx]
            actions.append(
                {
                    "_index": index_name,
                    "_id": document.id,
                    "_source": source,
                }
            )
        bulk(client.options(request_timeout=100), actions, raise_on_error=True)
        total += len(actions)
    return total


def deploy_to_elasticsearch(
    *,
    documents: Iterable[IndexDocument],
    es_host: str,
    index_name: str,
    embedding_model_name: str | None = None,
    embedding_dim: int | None = None,
    prompt_strategy: str = "none",
    overwrite: bool = False,
    batch_size: int = 200,
    embedding_batch_size: int = 16,
    max_text_chars: int = 32768,
    shards: int = 1,
    replicas: int = 0,
) -> int:
    client = build_elasticsearch_client(es_host)
    model = load_sentence_transformer(embedding_model_name) if embedding_model_name else None
    create_elasticsearch_index(
        client,
        index_name=index_name,
        embedding_dim=embedding_dim if model is not None else None,
        overwrite=overwrite,
        shards=shards,
        replicas=replicas,
    )
    return index_documents(
        client,
        index_name=index_name,
        documents=documents,
        embedding_model=model,
        prompt_strategy=prompt_strategy,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        max_text_chars=max_text_chars,
    )
