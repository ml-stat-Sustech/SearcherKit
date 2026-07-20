from __future__ import annotations

from dataclasses import dataclass, field

from searcherkit.interfaces.tui.ui.selection_types import ChatSelection

@dataclass
class TuiViewState:
    """Pure display state for the terminal UI."""

    show_thinking: bool = False
    show_tool_detail: bool = False
    chat_scroll_top: int | None = None
    chat_selection: ChatSelection | None = None
    selecting: bool = False
    query_history: list[str] = field(default_factory=list)
    history_index: int | None = None
    saved_input: str = ""
    spinner_frame: int = 0
