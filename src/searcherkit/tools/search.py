"""Native search and visit tools backed by configured data sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import overload
import asyncio

from searcherkit.common.errors import RecoverableError
from searcherkit.sources import DataSource, SearchResult, SourceError, build_source
from searcherkit.tools.base import BaseTool, ToolConfig
from searcherkit.tools.summarizer import Summarizer


def _search_extensions(
    *,
    queries: list[str],
    result_groups: list[list[SearchResult]],
    source: str | None = None,
) -> dict[str, object]:
    """Build tool extensions for search hits.

    ``documents`` keeps per-hit identity/title/url and the query that produced
    the hit (and optional source). ``searched_ids`` remains a flat id list for
    existing callers.
    """
    documents: list[dict[str, object]] = []
    for query, group in zip(queries, result_groups, strict=True):
        for result in group:
            item: dict[str, object] = {
                "id": result.document.id,
                "title": result.document.title,
                "url": result.document.url,
                "query": query,
            }
            if source is not None:
                item["source"] = source
            documents.append(item)
    return {
        "searched_ids": [document["id"] for document in documents],
        "documents": documents,
    }


def _format_results(results: list[SearchResult]) -> str:
    text = ""
    for i, result in enumerate(results, start=1):
        text += f"{i}. [{result.document.title}]({result.document.url})\n"
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
        argument_mapping: Mapping[str, str] | None = None,
        response_char_limit: int | None = None,
        summarizer: Summarizer | None = None,
        summary_goal_key = "query",
    ) -> None: ...

    def __init__(
        self,
        source: DataSource | None = None,
        *,
        name: str = "search",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        argument_mapping: Mapping[str, str] | None = None,
        response_char_limit: int | None = None,
        summarizer: Summarizer | None = None,
        summary_goal_key = "query",
        config: ToolConfig | None = None,
    ) -> None:
        if config:
            if not config.source:
                raise ValueError("SearchTool requires a source to be created from a tool config")
            if len(config.source) != 1:
                raise ValueError("SearchTool requires exactly one source")
            self.__init__(
                build_source(config.source[0]),
                name = config.name,
                description=config.description,
                inputSchema=config.inputSchema or inputSchema,
                argument_mapping=config.argument_mapping or argument_mapping,
                response_char_limit=config.response_char_limit,
                summarizer=Summarizer(config=config.summarizer) if config.summarizer else None,
                summary_goal_key=config.summary_goal_key,
            )
            return
        if source is None:
            raise ValueError("SearchTool requires a source")
        super().__init__(
            name=name,
            description=description,
            inputSchema=inputSchema,
            argument_mapping=argument_mapping,
            summarizer=summarizer,
            summary_goal_key=summary_goal_key,
        )
        self.source = source
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, *, query: str | list[str], top_k: int = 10) -> tuple[str, dict[str, object]]:
        """Search the configured data source."""
        try:
            if isinstance(query, list):
                queries = list(query)
                results = await asyncio.gather(
                    *[self.source.search(q, top_k=top_k) for q in queries],
                    return_exceptions=False,
                )
            else:
                queries = [query]
                results = [await self.source.search(query, top_k=top_k)]
        except SourceError as e:
            raise RecoverableError(str(e)) from e
        ret = []
        for r in results:
            text = _format_results(r)
            if self.response_char_limit is not None:
                text = _limit_response(text, self.response_char_limit)
            ret.append(text)
        return "\n=======\n".join(ret), _search_extensions(
            queries=queries,
            result_groups=list(results),
        )

    async def close(self) -> None:
        await self.source.close()
