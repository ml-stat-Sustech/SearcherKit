"""Provider-neutral interfaces for searchable data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(slots=True)
class Document:
    id: str
    text: str
    title: str | None = None
    url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    document: Document
    score: float | None = None
    snippet: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    """Minimal contract every searchable data source adapter should implement."""

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        """Return ranked documents for a natural-language query."""

    async def fetch(self, document_id: str) -> Document:
        """Return a full document by id."""
