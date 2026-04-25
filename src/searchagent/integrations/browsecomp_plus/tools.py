"""BrowseComp Plus MCP tool adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from fastmcp.client.client import CallToolResult

from searchagent.tools.mcp import BaseMCPTool, ToolFatalError, ToolRecoverableError


class BrowseCompPlusSearch(BaseMCPTool):
    """BrowseComp Plus search formatted like Tongyi DeepResearch web results."""

    def __init__(
        self,
        mcp_tool_name: str,
        endpoint: str,
        *,
        name: str = "search",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            mcp_tool_name=mcp_tool_name,
            endpoint=endpoint,
            **kwargs,
        )

    async def _run(self, **kwargs: Any) -> str:
        query = kwargs.get("query")
        if not isinstance(query, str):
            raise ToolFatalError("BrowseCompPlusSearch requires a string query")
        result = await self._run_mcp_tool(kwargs)
        return self._format_results(result, query)

    def _format_results(self, result: CallToolResult, query: str) -> str:
        content = getattr(result, "content", None)
        if not isinstance(content, list) or not content:
            raise ToolRecoverableError("BrowseComp Plus search returned no content")
        raw_text = getattr(content[0], "text", None)
        if not isinstance(raw_text, str):
            raise ToolRecoverableError("BrowseComp Plus search returned non-text content")

        try:
            results = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ToolRecoverableError("BrowseComp Plus search returned invalid JSON") from exc
        if not isinstance(results, list):
            raise ToolRecoverableError("BrowseComp Plus search JSON must be a list")

        formatted = []
        for idx, item in enumerate(results, 1):
            if not isinstance(item, Mapping):
                raise ToolRecoverableError("BrowseComp Plus search item must be an object")
            snippet = item.get("snippet")
            if not isinstance(snippet, str):
                raise ToolRecoverableError("BrowseComp Plus search item is missing snippet")

            title = self._title_from_snippet(snippet)
            formatted.append(f"{idx}. [{title}]\n{snippet}")

        return (
            f"A search for '{query}' found {len(formatted)} results:\n\n"
            "## Web Results\n"
            + "\n\n".join(formatted)
        )

    @staticmethod
    def _title_from_snippet(snippet: str) -> str:
        if snippet.startswith("---\ntitle:"):
            lines = snippet.split("\n")
            if len(lines) > 1:
                title = lines[1].replace("title:", "").strip().strip('"')
                if title:
                    return title

        first_line = snippet.split("\n")[0].strip()
        return first_line[:50] + "..." if len(first_line) > 50 else first_line
