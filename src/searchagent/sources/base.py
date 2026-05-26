"""Provider-neutral interfaces for searchable data sources."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from searchagent.common.retry import RetryConfig


@dataclass(slots=True)
class Document:
    id: str
    text: str
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    document: Document
    score: float | None = None
    snippet: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class DataSource(ABC):
    """Minimal contract every searchable data source adapter should implement."""
    def __init__(self, *, config: SourceConfig) -> None:
        ...

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        """Return ranked documents for a natural-language query."""
        ...

    async def fetch(self, document_id: str, *, goal: str | None = None) -> Document:
        """Return a full document by id."""
        ...


@dataclass
class SourceConfig:
    """
    The *type* discriminator selects the concrete source implementation.
    Only fields relevant to the selected type need to be populated.

    When *type* is ``"custom"``, *target* must be a ``pkg://`` import path
    pointing to a class that satisfies the :class:`DataSource` contract.
    """

    type: str = ""
    name: str = ""
    target: str | None = None

    # -- Elasticsearch -------------------------------------------------------
    hosts: Any = None
    index: str | None = None
    search_fields: list[str] | None = None
    title_field: str = "title"
    text_field: str = "text"
    url_field: str = "url"
    document_id_field: str | None = None
    fetch_field: str | None = None
    metadata_fields: list[str] | None = None
    highlight: bool = True
    highlight_fragment_size: int = 256
    snippet_chars: int = 512
    request_timeout: float | None = None
    es_max_concurrency: int | None = None
    client_kwargs: dict[str, Any] | None = None
    vector_search_mode: str = "bm25"
    vector_field: str = "text_vector"
    embedding_prefix: str = ""
    embedding_model: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_timeout: float = 60
    embedding_max_concurrency: int | None = None
    embedding_default_kwargs: dict[str, Any] | None = None
    embedding_retry_config: RetryConfig | None = None
    search_summary_enabled: bool = False
    fetch_summary_enabled: bool = False
    summary_model: str | None = None
    summary_api_key: str | None = None
    summary_base_url: str | None = None
    summary_max_chars: int = 400000
    summary_timeout: float = 3600
    summary_max_concurrency: int | None = None
    summary_default_kwargs: dict[str, Any] | None = None
    summary_retry_config: RetryConfig | None = None

    # -- Memory --------------------------------------------------------------
    documents: list[Document] | None = None
