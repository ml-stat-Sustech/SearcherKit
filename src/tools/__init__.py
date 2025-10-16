"""Tool registries for different agents."""

from .visit_multi_turn import VisitPage
from .search import SearchTool
from .visit_single_turn import VisitTool

WEBWALKER_TOOLS = {VisitPage.name: VisitPage()}
WEBDANCER_TOOLS = {
    SearchTool.name: SearchTool(),
    VisitTool.name: VisitTool(),
}

__all__ = [
    "VisitPage",
    "SearchTool",
    "VisitTool",
    "WEBWALKER_TOOLS",
    "WEBDANCER_TOOLS",
]
