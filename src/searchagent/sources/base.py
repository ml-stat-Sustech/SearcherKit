"""Provider-neutral interfaces for searchable data sources."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from searchagent.common.errors import SearchAgentError
from searchagent.common.retry import RetryConfig


class SourceError(SearchAgentError):
    """Data source or document retrieval failure."""


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

    async def close(self) -> None:
        """Release source resources."""


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

    # -- Web ----------------------------------------------------------------
    serper_api_key: str | None = None
    serper_endpoint: str = "https://google.serper.dev/search"
    serper_timeout: float = 30
    jina_api_key: str | None = None
    jina_base_url: str = "https://r.jina.ai/"
    jina_timeout: float = 60
    web_max_concurrency: int | None = None
    web_retry_config: RetryConfig | None = None

    # -- Memory --------------------------------------------------------------
    documents: list[Document] | None = None

    # -- Local file ----------------------------------------------------------
    root_path: str | None = None
    local_file_encoding: str = "utf-8"
    local_file_errors: str = "replace"
