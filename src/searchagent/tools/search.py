"""Native search and visit tools backed by configured data sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, overload

from searchagent.sources import DataSource, Document, SearchResult, build_source
from searchagent.tools.base import BaseTool, ToolConfig


SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1},
    },
    "required": ["query"],
    "additionalProperties": False,
}

VISIT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_id": {"type": "string"},
    },
    "required": ["document_id"],
    "additionalProperties": False,
}


def _build_source_from_config(
    config: ToolConfig,
    *,
    sources: Mapping[str, DataSource] | None = None,
) -> DataSource:
    if config.source:
        if sources is None or config.source not in sources:
            available = sorted(sources.keys()) if sources else []
            raise ValueError(
                f"Tool {config.name!r} references unknown source "
                f"{config.source!r}; available sources: {available}"
            )
        return sources[config.source]

    source = config.extra.get("source")
    if source is not None:
        return source

    source_config = config.source_config or config.extra.get("source_config")
    if source_config is not None:
        return build_source(source_config)

    raise ValueError(
        f"Tool {config.name!r} requires source, source_config, "
        "extra.source, or extra.source_config"
    )


def _document_payload(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "title": document.title,
        "url": document.url,
        "text": document.text,
        "metadata": dict(document.metadata),
    }


def _document_reference_payload(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "title": document.title,
        "url": document.url,
        "metadata": dict(document.metadata),
    }


def _result_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "document": _document_reference_payload(result.document),
        "score": result.score,
        "snippet": result.snippet,
        "metadata": dict(result.metadata),
    }


def _limit_response(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Truncated] Response exceeded configured content limit."


class SearchTool(BaseTool):
    """Search a configured data source."""

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        source: DataSource,
        *,
        name: str = "search",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        source: DataSource | None = None,
        *,
        name: str = "search",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
        config: ToolConfig | None = None,
        sources: Mapping[str, DataSource] | None = None,
    ) -> None:
        if config is not None:
            source = _build_source_from_config(config, sources=sources)
            if config.inputSchema is None:
                config.inputSchema = SEARCH_INPUT_SCHEMA
            if config.description is None:
                config.description = "Search the configured data source."
            super().__init__(config=config)
        else:
            if source is None:
                raise ValueError("SearchTool requires a source")
            super().__init__(
                name=name,
                description=description or "Search the configured data source.",
                inputSchema=inputSchema or SEARCH_INPUT_SCHEMA,
            )
        self.source = source
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, **kwargs: Any) -> str:
        query = str(kwargs["query"])
        top_k = int(kwargs.get("top_k", 10))
        results = await self.source.search(query, top_k=top_k)
        text = json.dumps(
            [_result_payload(result) for result in results],
            ensure_ascii=False,
        )
        return _limit_response(text, self.response_char_limit)


class VisitTool(BaseTool):
    """Fetch one document from a configured data source."""

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        source: DataSource,
        *,
        name: str = "visit",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        source: DataSource | None = None,
        *,
        name: str = "visit",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
        config: ToolConfig | None = None,
        sources: Mapping[str, DataSource] | None = None,
    ) -> None:
        if config is not None:
            source = _build_source_from_config(config, sources=sources)
            if config.inputSchema is None:
                config.inputSchema = VISIT_INPUT_SCHEMA
            if config.description is None:
                config.description = "Fetch a document from the configured data source."
            super().__init__(config=config)
        else:
            if source is None:
                raise ValueError("VisitTool requires a source")
            super().__init__(
                name=name,
                description=description or "Fetch a document from the configured data source.",
                inputSchema=inputSchema or VISIT_INPUT_SCHEMA,
            )
        self.source = source
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, **kwargs: Any) -> str:
        document_id = str(kwargs["document_id"])
        document = await self.source.fetch(document_id)
        return _limit_response(
            json.dumps(_document_payload(document), ensure_ascii=False),
            self.response_char_limit,
        )
