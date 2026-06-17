"""Native multi-source search tool backed by configured data sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload

from searchagent.errors import RecoverableError, SourceError
from searchagent.sources import DataSource, SearchResult, build_source
from searchagent.tools.base import BaseTool, ToolConfig

SEARCH_DESCRIPTION = """
Search the configured data sources.
Choose source to search from the list.
{sources}
"""

SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        # "top_k": {"type": "integer", "minimum": 1},
        "source": {"type": "string"},
    },
    "required": ["query", "source"],
    "additionalProperties": False,
}

def _format_results(results: list[SearchResult]) -> str:
    text = ""
    for i, result in enumerate(results, start=1):
        # text += f"{i}. [{result.document.title}]({result.document.url})\n"
        text += f"{result.snippet or result.document.text}\n"
        # text += f"Score: {result.score:.2f}\n"
        # if result.document.metadata:
        #     text += f"Metadata: {json.dumps(result.document.metadata, ensure_ascii=False, indent=None)}\n"
        text += "\n"

    return text

def _limit_response(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Truncated] Response exceeded configured content limit."


class MultiSourceSearchTool(BaseTool):
    """Search a configured data source."""

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        source_map: dict[str, DataSource],
        *,
        name: str = "search",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        source_map: dict[str, DataSource] | None = None,
        *,
        name: str = "search",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
        config: ToolConfig | None = None,
    ) -> None:
        if config:
            if not getattr(config, "source", None):
                raise ValueError("MultiSourceSearchTool requires a list of sources to be created from a tool config")
            source_names = getattr(config, "source")
            if not isinstance(source_names, list):
                raise ValueError("MultiSourceSearchTool requires a list of sources")
            self.__init__(
                {s: build_source(s) for s in source_names},
                name=config.name,
                description=config.description,
                inputSchema=config.inputSchema or inputSchema,
                response_char_limit=config.response_char_limit,
            )
            return
        
        if not source_map:
            raise ValueError("MultiSourceSearchTool requires a dict of sources")
        source_names = sorted(source_map.keys())
        rendered_description = description or SEARCH_DESCRIPTION.format(
            sources="\n".join(name for name in source_names)
            )
        super().__init__(
            name=name,
            description=rendered_description,
            inputSchema=inputSchema or SEARCH_INPUT_SCHEMA,
        )
        self.source_map = source_map
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, **kwargs: Any) -> str:
        query = str(kwargs["query"])
        top_k = int(kwargs.get("top_k", 5))
        source_name = str(kwargs["source"])
        source = self.source_map.get(source_name)
        if source is None:
            available_sources = ", ".join(sorted(self.source_map.keys()))
            raise RecoverableError(
                f"unknown source {source_name!r}; available sources: {available_sources}"
            )
        try:
            results = await source.search(query, top_k=top_k)
        except SourceError as e:
            raise RecoverableError(str(e)) from e
        text = _format_results(results)
        if self.response_char_limit is not None:
            text = _limit_response(text, self.response_char_limit)
        return text

    async def close(self) -> None:
        for source in self.source_map.values():
            await source.close()
