"""Elasticsearch-backed data source adapters."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import nullcontext
from collections.abc import Mapping, Sequence
from typing import Any, overload

import json_repair

from searchagent.common.retry import RetryConfig, RetryPolicy, retry_async
from searchagent.errors import SourceError
from searchagent.log import get_logger

from .base import Document, SearchResult, SourceConfig, DataSource

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch import (
        ApiError,
        BadRequestError,
        ConflictError,
        NotFoundError,
        SerializationError,
        TransportError,
    )
except ImportError:
    AsyncElasticsearch = None  # type: ignore[assignment]
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

_OPENAI_ERRORS: tuple[type[Exception], ...] = ()
SUMMARY_FALLBACK_CHARS = 4096
logger = get_logger(__name__)


def _lazy_import_openai_errors() -> tuple[type[Exception], ...]:
    global _OPENAI_ERRORS
    if _OPENAI_ERRORS:
        return _OPENAI_ERRORS
    from openai import APIError, APIConnectionError, APITimeoutError, OpenAIError, RateLimitError

    _OPENAI_ERRORS = (
        OpenAIError,
        APIError,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
    return _OPENAI_ERRORS


SUMMARY_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

## Task Guidelines
1. Locate the portions that directly support the goal.
2. Extract the most relevant evidence.
3. Provide a concise summary.

Respond strictly in JSON:
{{
  "evidence": "...",
  "summary": "..."
}}

## User Goal
{goal}

## Webpage Content
{webpage_content}
"""


class SummaryError(SourceError):
    """Raised when the summary model response is missing or malformed."""


class EmbeddingError(SourceError):
    """Raised when the embedding model response is missing or malformed."""


def _validate_max_concurrency(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")


def _build_semaphore(max_concurrency: int | None) -> asyncio.Semaphore | None:
    return asyncio.Semaphore(max_concurrency) if max_concurrency else None


class ElasticsearchSource(DataSource):
    """Search and fetch documents from an Elasticsearch index.

    This source expects documents to use common web-corpus fields by default:
    ``title``, ``text``, ``url``, and optional ``links``. Field names can be
    overridden for other indexes.
    """

    @overload
    def __init__(
        self,
        *,
        config: SourceConfig,
        client: Any | None = None,
    ) -> None: ...

    @overload
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
        es_max_concurrency: int | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        vector_search_mode: str = "bm25",
        vector_field: str = "text_vector",
        embedding_prefix: str = "",
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_timeout: float = 60,
        embedding_max_concurrency: int | None = None,
        embedding_default_kwargs: Mapping[str, Any] | None = None,
        embedding_retry_config: RetryConfig | None = None,
        embedding_client: Any | None = None,
        search_summary_enabled: bool = False,
        fetch_summary_enabled: bool = False,
        summary_model: str | None = None,
        summary_api_key: str | None = None,
        summary_base_url: str | None = None,
        summary_max_chars: int = 400000,
        summary_timeout: float = 3600,
        summary_max_concurrency: int | None = None,
        summary_default_kwargs: Mapping[str, Any] | None = None,
        summary_retry_config: RetryConfig | None = None,
        summary_client: Any | None = None,
    ) -> None: ...

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
        es_max_concurrency: int | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        vector_search_mode: str = "bm25",
        vector_field: str = "text_vector",
        embedding_prefix: str = "",
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_timeout: float = 60,
        embedding_max_concurrency: int | None = None,
        embedding_default_kwargs: Mapping[str, Any] | None = None,
        embedding_retry_config: RetryConfig | None = None,
        embedding_client: Any | None = None,
        search_summary_enabled: bool = False,
        fetch_summary_enabled: bool = False,
        summary_model: str | None = None,
        summary_api_key: str | None = None,
        summary_base_url: str | None = None,
        summary_max_chars: int = 400000,
        summary_timeout: float = 3600,
        summary_max_concurrency: int | None = None,
        summary_default_kwargs: Mapping[str, Any] | None = None,
        summary_retry_config: RetryConfig | None = None,
        summary_client: Any | None = None,
        config: SourceConfig | None = None,
    ) -> None:
        if config is not None:
            hosts = hosts or config.hosts
            index = index or (config.index or "")
            search_fields = search_fields or config.search_fields
            title_field = config.title_field
            text_field = config.text_field
            url_field = config.url_field
            document_id_field = document_id_field or config.document_id_field
            fetch_field = fetch_field or config.fetch_field
            metadata_fields = metadata_fields or config.metadata_fields
            highlight = config.highlight
            highlight_fragment_size = config.highlight_fragment_size
            snippet_chars = config.snippet_chars
            request_timeout = request_timeout or config.request_timeout
            if es_max_concurrency is None:
                es_max_concurrency = config.es_max_concurrency
            client_kwargs = client_kwargs or config.client_kwargs
            vector_search_mode = config.vector_search_mode
            vector_field = config.vector_field
            embedding_prefix = config.embedding_prefix
            embedding_model = embedding_model or config.embedding_model
            embedding_api_key = embedding_api_key or config.embedding_api_key
            embedding_base_url = embedding_base_url or config.embedding_base_url
            embedding_timeout = config.embedding_timeout
            if embedding_max_concurrency is None:
                embedding_max_concurrency = config.embedding_max_concurrency
            embedding_default_kwargs = embedding_default_kwargs or config.embedding_default_kwargs
            embedding_retry_config = embedding_retry_config or config.embedding_retry_config
            search_summary_enabled = config.search_summary_enabled
            fetch_summary_enabled = config.fetch_summary_enabled
            summary_model = summary_model or config.summary_model
            summary_api_key = summary_api_key or config.summary_api_key
            summary_base_url = summary_base_url or config.summary_base_url
            summary_max_chars = config.summary_max_chars
            summary_timeout = config.summary_timeout
            if summary_max_concurrency is None:
                summary_max_concurrency = config.summary_max_concurrency
            summary_default_kwargs = summary_default_kwargs or config.summary_default_kwargs
            summary_retry_config = summary_retry_config or config.summary_retry_config

        if not index:
            raise ValueError("ElasticsearchSource requires index")
        if client is None and not hosts:
            raise ValueError("ElasticsearchSource requires hosts or client")
        if highlight_fragment_size <= 0:
            raise ValueError("highlight_fragment_size must be positive")
        if snippet_chars <= 0:
            raise ValueError("snippet_chars must be positive")
        if summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be positive")
        if summary_timeout <= 0:
            raise ValueError("summary_timeout must be positive")
        _validate_max_concurrency("es_max_concurrency", es_max_concurrency)
        _validate_max_concurrency("embedding_max_concurrency", embedding_max_concurrency)
        _validate_max_concurrency("summary_max_concurrency", summary_max_concurrency)
        if vector_search_mode not in {"bm25", "hybrid", "vector"}:
            raise ValueError("vector_search_mode must be 'bm25', 'hybrid', or 'vector'")
        uses_vector_search = vector_search_mode in {"hybrid", "vector"}
        if highlight and vector_search_mode not in {"bm25", "hybrid"}:
            raise ValueError("highlight requires vector_search_mode to be 'bm25' or 'hybrid'")
        if uses_vector_search and not vector_field:
            raise ValueError("vector_field is required when vector search is enabled")
        if uses_vector_search and not embedding_model:
            raise ValueError("embedding_model is required when vector search is enabled")
        if uses_vector_search and embedding_client is None and not embedding_api_key:
            raise ValueError("embedding_api_key is required when vector search is enabled")
        if embedding_timeout <= 0:
            raise ValueError("embedding_timeout must be positive")
        if uses_vector_search and embedding_client is None:
            from openai import AsyncOpenAI

            embedding_client_kwargs: dict[str, Any] = {"api_key": embedding_api_key}
            if embedding_base_url:
                embedding_client_kwargs["base_url"] = embedding_base_url
            embedding_client = AsyncOpenAI(**embedding_client_kwargs)
        any_summary_enabled = search_summary_enabled or fetch_summary_enabled
        if any_summary_enabled and not summary_model:
            raise ValueError("summary_model is required when summary is enabled")
        if any_summary_enabled and summary_client is None and not summary_api_key:
            raise ValueError("summary_api_key is required when summary is enabled")
        if any_summary_enabled and summary_client is None:
            from openai import AsyncOpenAI

            summary_client_kwargs: dict[str, Any] = {"api_key": summary_api_key}
            if summary_base_url:
                summary_client_kwargs["base_url"] = summary_base_url
            summary_client = AsyncOpenAI(**summary_client_kwargs)

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
        self._es_semaphore = _build_semaphore(es_max_concurrency)
        self.vector_search_mode = vector_search_mode
        self.vector_field = vector_field
        self.embedding_prefix = embedding_prefix
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.embedding_timeout = embedding_timeout
        self._embedding_semaphore = _build_semaphore(embedding_max_concurrency)
        self.embedding_default_kwargs = dict(embedding_default_kwargs or {})
        self.embedding_retry_policy = (
            RetryPolicy(config=embedding_retry_config)
            if embedding_retry_config is not None
            else RetryPolicy(exceptions=(*_lazy_import_openai_errors(), EmbeddingError, ValueError))
            if self._uses_vector_search
            else None
        )
        self._embedding_client = embedding_client
        self.search_summary_enabled = search_summary_enabled
        self.fetch_summary_enabled = fetch_summary_enabled
        self.summary_model = summary_model
        self.summary_api_key = summary_api_key
        self.summary_base_url = summary_base_url
        self.summary_max_chars = summary_max_chars
        self.summary_timeout = summary_timeout
        self._summary_semaphore = _build_semaphore(summary_max_concurrency)
        self.summary_default_kwargs = dict(summary_default_kwargs or {})
        self.summary_retry_policy = (
            RetryPolicy(config=summary_retry_config)
            if summary_retry_config is not None
            else RetryPolicy(
                exceptions=(
                    *_lazy_import_openai_errors(),
                    SummaryError,
                    ValueError,
                )
            )
            if any_summary_enabled
            else None
        )
        self._summary_client = summary_client
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
        kwargs = dict(client_kwargs)
        if self.request_timeout is not None:
            kwargs.setdefault("request_timeout", self.request_timeout)
        if not AsyncElasticsearch:
            raise ImportError(
                "AsyncElasticsearch client not available, use uv sync --extra elasticsearch-source to install dependency for Elasticsearch source"
            )
        return AsyncElasticsearch(hosts, **kwargs)

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def _limit_es_request(self) -> Any:
        if self._es_semaphore is None:
            return nullcontext()
        return self._es_semaphore

    def _limit_embedding_request(self) -> Any:
        if self._embedding_semaphore is None:
            return nullcontext()
        return self._embedding_semaphore

    def _limit_summary_request(self) -> Any:
        if self._summary_semaphore is None:
            return nullcontext()
        return self._summary_semaphore

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_vector = await self._query_vector(query) if self._uses_vector_search else None
        body = self._search_body(query=query, top_k=top_k, query_vector=query_vector)
        response = await self._call_search(body)
        hits = self._hits(response)
        results = [self._search_result_from_hit(hit) for hit in hits]
        if not self.search_summary_enabled:
            return results
        if not results:
            return []
        return [await self._summary_search_result(query=query, results=results)]

    async def fetch(self, document_id: str, *, goal: str | None = None) -> Document:
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
            document = self._document_from_hit(hits[0])
            if not self.fetch_summary_enabled:
                return document
            return await self._summary_document(document, goal=goal)

        try:
            async with self._limit_es_request():
                hit = await self.client.get(
                    index=self.index,
                    id=document_id,
                )
        except _ELASTICSEARCH_ERRORS as exc:
            raise SourceError(f"failed to fetch Elasticsearch document {document_id!r}") from exc
        document = self._document_from_hit(hit)
        if not self.fetch_summary_enabled:
            return document
        return await self._summary_document(document, goal=goal)

    def _search_body(
        self,
        *,
        query: str,
        top_k: int,
        query_vector: Sequence[float] | None = None,
    ) -> dict[str, Any]:
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
        if query_vector is not None:
            knn: dict[str, Any] = {
                "field": self.vector_field,
                "query_vector": list(query_vector),
                "k": top_k,
                "num_candidates": top_k,
            }
            body["knn"] = knn
            if self.vector_search_mode == "vector":
                body.pop("query")
        if self.highlight:
            body["highlight"] = {
                "fields": {
                    self.text_field: {"fragment_size": self.highlight_fragment_size},
                }
            }
        return body

    @property
    def _uses_vector_search(self) -> bool:
        return self.vector_search_mode in {"hybrid", "vector"}

    async def _query_vector(self, query: str) -> Sequence[float]:
        try:
            return await retry_async(
                self._request_query_vector,
                query,
                policy=self.embedding_retry_policy
                or RetryPolicy(exceptions=(*_lazy_import_openai_errors(), EmbeddingError, ValueError)),
                op_name="elasticsearch.embedding",
            )
        except (*_lazy_import_openai_errors(), EmbeddingError, ValueError) as exc:
            raise SourceError("failed to generate Elasticsearch query embedding") from exc

    async def _request_query_vector(self, query: str) -> Sequence[float]:
        if not self.embedding_model:
            raise ValueError("embedding_model is not configured")
        if self._embedding_client is None:
            raise ValueError("embedding client is not configured")
        payload: dict[str, Any] = {
            "model": self.embedding_model,
            "input": f"{self.embedding_prefix}{query}",
            "timeout": self.embedding_timeout,
            **self.embedding_default_kwargs,
        }
        async with self._limit_embedding_request():
            response = await self._embedding_client.embeddings.create(**payload)
        data = response.data if hasattr(response, "data") else []
        if not data:
            raise EmbeddingError("embedding model returned no vectors")
        embedding = data[0].embedding
        if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
            raise EmbeddingError("embedding model returned malformed vector")
        return [float(value) for value in embedding]

    async def _call_search(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            async with self._limit_es_request():
                return await self.client.search(
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

    async def _summary_document(self, document: Document, *, goal: str | None) -> Document:
        goal_label = goal or "N/A"
        content = self._format_document_content(document)
        try:
            evidence, summary = await self._summarize(
                goal=goal_label,
                content=content,
            )
        except SourceError as exc:
            logger.warning(
                "Elasticsearch visit summary failed; falling back to raw evidence document_id=%r goal=%r error=%s",
                document.id,
                goal_label,
                exc,
            )
            evidence = self._fallback_summary_evidence(content)
            summary = ""
        formatted = self._format_visit_summary(
            document.url or document.title or document.id,
            goal_label,
            evidence,
            summary,
        )
        return Document(
            id=document.id,
            title=document.title,
            text=formatted,
            url=document.url,
            metadata=document.metadata,
        )

    async def _summary_search_result(self, *, query: str, results: list[SearchResult]) -> SearchResult:
        content = "\n\n".join(
            self._format_document_content(result.document)
            for result in results
        )
        summary_fallback = False
        try:
            evidence, summary = await self._summarize(
                goal=query,
                content=content,
            )
        except SourceError as exc:
            logger.warning(
                "Elasticsearch search summary failed; falling back to raw evidence query=%r error=%s",
                query,
                exc,
            )
            evidence = self._fallback_summary_evidence(content)
            summary = ""
            summary_fallback = True
        formatted = self._format_search_summary(query, evidence, summary)
        document = Document(
            id=f"summary:{query}",
            title=f"Summary for {query}",
            text=formatted,
            url=None,
            metadata={
                "summary": True,
                "summary_fallback": summary_fallback,
                "result_count": len(results),
            },
        )
        return SearchResult(
            document=document,
            score=results[0].score,
            snippet=formatted,
            metadata={"summary": True, "summary_fallback": summary_fallback},
        )

    def _format_document_content(self, document: Document) -> str:
        title = document.title or "[No Title]"
        url = document.url or document.id
        return f"[{title}]({url})\n{document.text[:int(self.summary_max_chars / 5)]}".strip() # TODO: 当前默认top_k = 5, 按文章截断保证上下文限制，之后需要对top_k更好地处理

    def _fallback_summary_evidence(self, content: str) -> str:
        compact = "\n".join(line.strip() for line in content.splitlines() if line.strip())
        limit = min(self.summary_max_chars, SUMMARY_FALLBACK_CHARS)
        return compact[:limit] or "No evidence extracted."

    async def _summarize(self, *, goal: str, content: str) -> tuple[str, str]:
        try:
            return await retry_async(
                self._request_summary,
                goal,
                content[: self.summary_max_chars],
                policy=self.summary_retry_policy or RetryPolicy(exceptions=(SummaryError,)),
                op_name="elasticsearch.summary",
            )
        except (*_lazy_import_openai_errors(), SummaryError, ValueError) as exc:
            raise SourceError("failed to generate Elasticsearch summary") from exc

    async def _request_summary(self, goal: str, content: str) -> tuple[str, str]:
        if not self.summary_model:
            raise SummaryError("summary_model is not configured")
        prompt = SUMMARY_PROMPT.format(goal=goal or "N/A", webpage_content=content)
        payload: dict[str, Any] = {
            "model": self.summary_model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "timeout": self.summary_timeout,
            **self.summary_default_kwargs,
        }
        if self._summary_client is None:
            raise SummaryError("summary client is not configured")
        async with self._limit_summary_request():
            response = await self._summary_client.chat.completions.create(**payload)
        raw = response.choices[0].message.content if response.choices else None
        if not raw:
            raise SummaryError("summary model returned empty content")
        data = json_repair.loads(raw)
        if not isinstance(data, Mapping):
            raise SummaryError("summary model returned non-object JSON")
        evidence = str(data.get("evidence", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not evidence and not summary:
            raise SummaryError("summary model returned no evidence or summary")
        return evidence, summary

    @staticmethod
    def _format_search_summary(goal: str, evidence: str, summary: str) -> str:
        lines = [
            f"The useful information for query {goal} as follows:",
            "",
            "Evidence in page:",
            evidence or "No evidence extracted.",
            "",
            "Summary:",
            summary or "No summary available.",
            "",
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _format_visit_summary(source: str, goal: str, evidence: str, summary: str) -> str:
        lines = [
            f"The useful information in {source or 'Elasticsearch'} for user goal {goal or 'N/A'} as follows:",
            "",
            "Evidence in page:",
            evidence or "No evidence extracted.",
            "",
            "Summary:",
            summary or "No summary available.",
            "",
        ]
        return "\n".join(lines).strip()
