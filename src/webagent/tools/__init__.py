"""Tool registries for different agents."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

from .base import BaseTool, ToolCall, ToolResult

def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}

def use_local_wiki_tools() -> bool:
    return (
        _env_flag("WEBDANCER_USE_LOCAL_WIKI")
        or _env_flag("WEBWALKER_USE_LOCAL_WIKI")
        or _env_flag("RAG_USE_LOCAL_WIKI")
    )


@lru_cache(maxsize=1)
def build_webwalker_tools() -> Dict[str, BaseTool]:
    if use_local_wiki_tools():
        from .local_wiki import WebWalkerLocalWikiVisitTool

        visit_tool: BaseTool = WebWalkerLocalWikiVisitTool()
    else:
        from .visit_multi_turn import VisitPage

        visit_tool = VisitPage()

    return {visit_tool.name: visit_tool}


# @lru_cache(maxsize=1)
async def build_webdancer_tools() -> Dict[str, BaseTool]:

    if use_local_wiki_tools():
        from .local_wiki import LocalWikiSearchTool, LocalWikiVisitTool
        search_tool: BaseTool = LocalWikiSearchTool()
        visit_tool: BaseTool = LocalWikiVisitTool()
    else:
        # from .search import SearchTool
        # from .visit_single_turn import VisitTool
        # from .BrowseCompP import SearchTool, VisitTool
        from .BrowseCompP import FakeSearchTool as SearchTool
        from .BrowseCompP import FakeVisitTool as VisitTool
        search_tool = SearchTool()
        visit_tool = VisitTool()
        await search_tool.init("http://192.168.77.12:8100/mcp")
        await visit_tool.init("http://192.168.77.12:8100/mcp")

    tool_map = {
        search_tool.name: search_tool,
        visit_tool.name: visit_tool,
    }
    # Register friendly aliases expected by local_wiki prompts.
    # tool_map.setdefault("SearchLocalWiki", search_tool)
    # tool_map.setdefault("VisitForLocalWiki", visit_tool)
    return tool_map


@lru_cache(maxsize=1)
def build_rag_tools() -> Dict[str, BaseTool]:
    if use_local_wiki_tools():
        from .local_wiki import LocalWikiSearchTool, LocalWikiVisitTool

        search_tool: BaseTool = LocalWikiSearchTool()
        visit_tool: BaseTool = LocalWikiVisitTool()
        return {
            search_tool.name: search_tool,
            visit_tool.name: visit_tool,
        }

    from .search import SearchTool

    search_tool: BaseTool = SearchTool()
    return {search_tool.name: search_tool}


__all__ = [
    "BaseTool",
    "ToolCall",
    "ToolResult",
    "use_local_wiki_tools",
    "build_webwalker_tools",
    "build_webdancer_tools",
    "build_rag_tools",
]
