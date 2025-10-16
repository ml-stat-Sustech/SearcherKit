"""Tool registries for different agents."""

from .base import BaseTool, ToolCall, ToolResult
from .visit_multi_turn import VisitPage
from .search import SearchTool
from .visit_single_turn import VisitTool

WEBWALKER_TOOLS = {VisitPage.name: VisitPage()}
WEBDANCER_TOOLS = {
    SearchTool.name: SearchTool(),
    VisitTool.name: VisitTool(),
}

__all__ = [
    "BaseTool",
    "ToolCall",
    "ToolResult",
    "VisitPage",
    "SearchTool",
    "VisitTool",
    "WEBWALKER_TOOLS",
    "WEBDANCER_TOOLS",
]
