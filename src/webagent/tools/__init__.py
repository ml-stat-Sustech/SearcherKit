"""Tool registries for different agents."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from .base import BaseTool, ToolCall, ToolResult


def use_local_wiki_tools() -> bool:
    from .local_wiki import use_local_wiki_tools as _use_local_wiki_tools

    return _use_local_wiki_tools()


@lru_cache(maxsize=1)
def build_webwalker_tools() -> Dict[str, BaseTool]:
    from .visit_multi_turn import VisitPage

    return {VisitPage.name: VisitPage()}


@lru_cache(maxsize=1)
def build_webdancer_tools() -> Dict[str, BaseTool]:
    from .local_wiki import LocalWikiSearchTool, LocalWikiVisitTool
    from .search import SearchTool
    from .visit_single_turn import VisitTool

    if use_local_wiki_tools():
        search_tool: BaseTool = LocalWikiSearchTool()
        visit_tool: BaseTool = LocalWikiVisitTool()
    else:
        search_tool = SearchTool()
        visit_tool = VisitTool()

    tool_map = {
        search_tool.name: search_tool,
        visit_tool.name: visit_tool,
    }
    # Register friendly aliases expected by local_wiki prompts.
    tool_map.setdefault("SearchLocalWiki", search_tool)
    tool_map.setdefault("VisitForLocalWiki", visit_tool)
    return tool_map


__all__ = [
    "BaseTool",
    "ToolCall",
    "ToolResult",
    "use_local_wiki_tools",
    "build_webwalker_tools",
    "build_webdancer_tools",
]
