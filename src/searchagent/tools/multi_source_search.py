"""Native multi-source search tool backed by configured data sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import overload

from searchagent.common.errors import RecoverableError
from searchagent.sources import DataSource, SearchResult, SourceError, build_source
from searchagent.tools.base import BaseTool, ToolConfig

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
        argument_mapping: Mapping[str, str] | None = None,
        response_char_limit: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        source_map: dict[str, DataSource] | None = None,
        *,
        name: str = "search",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        argument_mapping: Mapping[str, str] | None = None,
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
                argument_mapping=config.argument_mapping or argument_mapping,
                response_char_limit=config.response_char_limit,
            )
            return
        
        if not source_map:
            raise ValueError("MultiSourceSearchTool requires a dict of sources")
        source_names = sorted(source_map.keys())
        super().__init__(
            name=name,
            description=description,
            inputSchema=inputSchema,
            argument_mapping=argument_mapping,
        )
        if description is None and self.description:
            sources = "\n".join(name for name in source_names)
            self.description = f"{self.description}\nChoose source to search from the list.\n{sources}"
        self.source_map = source_map
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, *, query: str, source: str, top_k: int = 5) -> str:
        """Search the configured data sources."""
        selected_source = self.source_map.get(source)
        if selected_source is None:
            available_sources = ", ".join(sorted(self.source_map.keys()))
            raise RecoverableError(
                f"unknown source {source!r}; available sources: {available_sources}"
            )
        try:
            results = await selected_source.search(query, top_k=top_k)
        except SourceError as e:
            raise RecoverableError(str(e)) from e
        text = _format_results(results)
        if self.response_char_limit is not None:
            text = _limit_response(text, self.response_char_limit)
        return text

    async def close(self) -> None:
        for source in self.source_map.values():
            await source.close()
