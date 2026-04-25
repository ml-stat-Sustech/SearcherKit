"""Elasticsearch-backed data source adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from searchagent.errors import SourceError

from .base import Document, SearchResult

try:
    from elasticsearch import Elasticsearch
    from elasticsearch import (
        ApiError,
        BadRequestError,
        ConflictError,
        NotFoundError,
        SerializationError,
        TransportError,
    )
except ImportError:
    Elasticsearch = None  # type: ignore[assignment]
    ApiError = BadRequestError = ConflictError = NotFoundError = SerializationError = TransportError = None  # type: ignore[assignment]


_ELASTICSEARCH_ERRORS = tuple(
    exc
    for exc in (
        ApiError,
        BadRequestError,
        ConflictError,
        NotFoundError,
        SerializationError,
        TransportError,
    )
    if isinstance(exc, type)
)


class ElasticsearchSource:
    """Search and fetch documents from an Elasticsearch index.

    This source expects documents to use common web-corpus fields by default:
    ``title``, ``text``, ``url``, and optional ``links``. Field names can be
    overridden for other indexes.
    """

    def __init__(
        self,
        hosts: str | Sequence[str] | None = None,
        index: str = "",
        *,
        client: Any | None = None,
        search_fields: Sequence[str] | None = None,
        title_field: str = "title",
        text_field: str = "text",
        url_field: str = "url",
        document_id_field: str | None = None,
        fetch_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        highlight: bool = True,
        highlight_fragment_size: int = 256,
        snippet_chars: int = 512,
        request_timeout: float | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not index:
            raise ValueError("ElasticsearchSource requires index")
        if client is None and not hosts:
            raise ValueError("ElasticsearchSource requires hosts or client")
        if highlight_fragment_size <= 0:
            raise ValueError("highlight_fragment_size must be positive")
        if snippet_chars <= 0:
            raise ValueError("snippet_chars must be positive")

        self.index = index
        self.search_fields = list(search_fields or [f"{title_field}^2", text_field])
        self.title_field = title_field
        self.text_field = text_field
        self.url_field = url_field
        self.document_id_field = document_id_field
        self.fetch_field = fetch_field or document_id_field
        self.metadata_fields = list(metadata_fields or ["links"])
        self.highlight = highlight
        self.highlight_fragment_size = highlight_fragment_size
        self.snippet_chars = snippet_chars
        self.request_timeout = request_timeout
        self.client = client or self._build_client(
            hosts=hosts,
            client_kwargs=client_kwargs or {},
        )

    def _build_client(
        self,
        *,
        hosts: str | Sequence[str] | None,
        client_kwargs: Mapping[str, Any],
    ) -> Any:
        if Elasticsearch is None:
            raise ImportError(
                "ElasticsearchSource requires the 'elasticsearch' package. "
                "Install with the elasticsearch or local-wiki extra."
            )
        kwargs = dict(client_kwargs)
        if self.request_timeout is not None:
            kwargs.setdefault("request_timeout", self.request_timeout)
        return Elasticsearch(hosts, **kwargs)

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        body = self._search_body(query=query, top_k=top_k)
        response = await self._call_search(body)
        hits = self._hits(response)
        return [self._search_result_from_hit(hit) for hit in hits]

    async def fetch(self, document_id: str) -> Document:
        if not document_id:
            raise ValueError("document_id must be non-empty")

        if self.fetch_field:
            body = {
                "size": 1,
                "query": {"term": {self.fetch_field: {"value": document_id}}},
            }
            response = await self._call_search(body)
            hits = self._hits(response)
            if not hits:
                raise KeyError(f"document not found: {document_id}")
            return self._document_from_hit(hits[0])

        try:
            hit = await asyncio.to_thread(
                self.client.get,
                index=self.index,
                id=document_id,
            )
        except _ELASTICSEARCH_ERRORS as exc:
            raise SourceError(f"failed to fetch Elasticsearch document {document_id!r}") from exc
        return self._document_from_hit(hit)

    def _search_body(self, *, query: str, top_k: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self.search_fields,
                    "type": "best_fields",
                }
            },
        }
        if self.highlight:
            body["highlight"] = {
                "fields": {
                    self.text_field: {"fragment_size": self.highlight_fragment_size},
                }
            }
        return body

    async def _call_search(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return await asyncio.to_thread(
                self.client.search,
                index=self.index,
                body=dict(body),
            )
        except _ELASTICSEARCH_ERRORS as exc:
            raise SourceError(f"failed to search Elasticsearch index {self.index!r}") from exc

    @staticmethod
    def _hits(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        hits = response.get("hits", {})
        if not isinstance(hits, Mapping):
            return []
        raw_hits = hits.get("hits", [])
        if not isinstance(raw_hits, list):
            return []
        return [hit for hit in raw_hits if isinstance(hit, Mapping)]

    def _search_result_from_hit(self, hit: Mapping[str, Any]) -> SearchResult:
        return SearchResult(
            document=self._document_from_hit(hit),
            score=self._score(hit),
            snippet=self._snippet(hit),
            metadata={"index": hit.get("_index")},
        )

    def _document_from_hit(self, hit: Mapping[str, Any]) -> Document:
        source = self._source(hit)
        fallback_id = str(hit.get("_id", ""))
        document_id = self._field_value(source, self.document_id_field) or fallback_id
        return Document(
            id=str(document_id),
            title=self._optional_str(source.get(self.title_field)),
            text=str(source.get(self.text_field, "") or ""),
            url=self._optional_str(source.get(self.url_field)),
            metadata=self._metadata(source),
        )

    @staticmethod
    def _source(hit: Mapping[str, Any]) -> Mapping[str, Any]:
        source = hit.get("_source", {})
        return source if isinstance(source, Mapping) else {}

    @staticmethod
    def _score(hit: Mapping[str, Any]) -> float | None:
        score = hit.get("_score")
        if isinstance(score, (int, float)):
            return float(score)
        return None

    def _snippet(self, hit: Mapping[str, Any]) -> str | None:
        highlight = hit.get("highlight")
        if isinstance(highlight, Mapping):
            fragments = highlight.get(self.text_field)
            if isinstance(fragments, list):
                text_fragments = [fragment for fragment in fragments if isinstance(fragment, str)]
                if text_fragments:
                    return "\n...\n".join(text_fragments).replace("<em>", "").replace("</em>", "")

        text = str(self._source(hit).get(self.text_field, "") or "")
        if not text:
            return None
        return text[: self.snippet_chars]

    def _metadata(self, source: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {}
        for field in self.metadata_fields:
            if field in source:
                metadata[field] = source[field]
        return metadata

    @staticmethod
    def _field_value(source: Mapping[str, Any], field: str | None) -> Any | None:
        if not field:
            return None
        return source.get(field)

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)
