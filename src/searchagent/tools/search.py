"""Native search and visit tools backed by configured data sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, overload

from searchagent.errors import RecoverableError, SourceError
from searchagent.sources import DataSource, SearchResult, build_source
from searchagent.tools.base import BaseTool, ToolConfig

SEARCH_DESCRIPTION = """
Search the configured data source.
"""

SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query_list": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "top_k": {"type": "integer", "minimum": 1},
    },
    "required": ["query_list"],
    "additionalProperties": False,
}

RESULT_SEPARATOR = "\n-*-*-\n"
DEFAULT_TOP_K = 3

def _format_results(results: list[SearchResult]) -> str:
    text = ""
    for i, result in enumerate(results, start=1):
        text += f"{i}. [{result.document.title}]({result.document.url})\n"
        text += f"{result.snippet or result.document.text}\n"
        text += f"Score: {result.score:.2f}\n"
        if result.document.metadata:
            text += f"Metadata: {json.dumps(result.document.metadata, ensure_ascii=False, indent=None)}\n"
        text += "\n"

    return text

def _limit_response(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Truncated] Response exceeded configured content limit."

def _coerce_query_list(kwargs: Mapping[str, Any]) -> list[str]:
    if "query_list" in kwargs:
        value = kwargs["query_list"]
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(query) for query in value]
        return [str(value)]
    if "query" in kwargs:
        return [str(kwargs["query"])]
    raise KeyError("query_list")


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
    ) -> None:
        if config:
            if not getattr(config, "source", None):
                raise ValueError("SearchTool requires a source to be created from a tool config")
            self.__init__(
                build_source(config.source),
                name=config.name,
                description=config.description,
                inputSchema=inputSchema or config.inputSchema,
                response_char_limit=config.response_char_limit,
            )
            return
        if source is None:
            raise ValueError("SearchTool requires a source")
        super().__init__(
            name=name,
            description=description or SEARCH_DESCRIPTION,
            inputSchema=inputSchema or SEARCH_INPUT_SCHEMA,
        )
        self.source = source
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, **kwargs: Any) -> str:
        query_list = _coerce_query_list(kwargs)
        top_k = int(kwargs.get("top_k", DEFAULT_TOP_K))
        chunks = []
        for query in query_list:
            try:
                results = await self.source.search(query, top_k=top_k)
            except SourceError as exc:
                raise RecoverableError(str(exc)) from exc
            chunks.append(_format_results(results))
        text = RESULT_SEPARATOR.join(chunks)
        if self.response_char_limit is not None:
            text = _limit_response(text, self.response_char_limit)
        return text

    async def close(self) -> None:
        await self.source.close()
