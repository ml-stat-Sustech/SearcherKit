"""In-memory source useful for examples and contract tests."""

from __future__ import annotations

from typing import overload

from .base import Document, SearchResult, SourceConfig, DataSource


class MemorySource(DataSource):
    """In-memory data source for testing and simple workloads."""

    @overload
    def __init__(self, *, config: SourceConfig) -> None: ...

    @overload
    def __init__(self, *, documents: list[Document] | None = None) -> None: ...

    def __init__(
        self,
        *,
        documents: list[Document] | None = None,
        config: SourceConfig | None = None,
    ) -> None:
        if config is not None:
            documents = config.documents or []

        self.documents = [
            doc if isinstance(doc, Document) else Document(**doc)
            for doc in (documents or [])
        ]

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        terms = {term.lower() for term in query.split() if term}
        scored: list[SearchResult] = []
        for document in self.documents:
            haystack = " ".join(
                part for part in [document.title or "", document.text] if part
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append(SearchResult(document=document, score=float(score)))
        scored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored[:top_k]

    async def fetch(self, document_id: str) -> Document:
        for document in self.documents:
            if document.id == document_id:
                return document
        raise KeyError(f"document not found: {document_id}")
