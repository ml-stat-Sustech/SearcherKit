"""Mouse/keyboard text selection in the chat viewport."""

from __future__ import annotations

from collections.abc import Callable

from searchagent.interfaces.tui.ui.formatting import _char_display_width
from searchagent.interfaces.tui.ui.selection_types import ChatPoint, ChatSelection
from searchagent.interfaces.tui.ui.view_state import TuiViewState


class SelectionManager:
    """Manages mouse/keyboard text selection in the chat viewport."""

    def __init__(
        self,
        view_state: TuiViewState,
        *,
        get_plain_lines: Callable[[], list[str]],
        get_view_height: Callable[[], int],
        get_scroll_top: Callable[..., int],
    ) -> None:
        self._view_state = view_state
        self._get_plain_lines = get_plain_lines
        self._get_view_height = get_view_height
        self._get_scroll_top = get_scroll_top

    def start(self, viewport_row: int, col: int) -> None:
        point = self._point_from_viewport(viewport_row, col)
        self._view_state.chat_selection = ChatSelection(anchor=point, head=point)
        self._view_state.selecting = True

    def update(self, viewport_row: int, col: int) -> None:
        if not self._view_state.selecting or self._view_state.chat_selection is None:
            return
        self._view_state.chat_selection.head = self._point_from_viewport(viewport_row, col)

    def finish(self, viewport_row: int, col: int) -> tuple[bool, str]:
        """Finish selection and return (has_text, selected_text)."""
        if not self._view_state.selecting or self._view_state.chat_selection is None:
            return False, ""
        self._view_state.chat_selection.head = self._point_from_viewport(viewport_row, col)
        self._view_state.selecting = False
        text = self.text()
        if not text:
            self.clear()
            return False, ""
        return True, text

    def clear(self) -> None:
        self._view_state.chat_selection = None
        self._view_state.selecting = False

    def text(self) -> str:
        bounds = self.bounds()
        if bounds is None:
            return ""
        start, end = bounds
        if start.row == end.row and start.col == end.col:
            return ""
        lines = self._get_plain_lines()
        if not lines:
            return ""
        start_row = min(max(start.row, 0), len(lines) - 1)
        end_row = min(max(end.row, 0), len(lines) - 1)
        selected: list[str] = []
        for row in range(start_row, end_row + 1):
            line = lines[row]
            left = start.col if row == start_row else 0
            right = end.col if row == end_row else len(line)
            selected.append(line[min(left, len(line)) : min(max(right, 0), len(line))].rstrip())
        return "\n".join(selected).strip("\n")

    def bounds(self) -> tuple[ChatPoint, ChatPoint] | None:
        selection = self._view_state.chat_selection
        if selection is None:
            return None
        start = selection.anchor
        end = selection.head
        if (end.row, end.col) < (start.row, start.col):
            start, end = end, start
        return start, end

    def _point_from_viewport(self, viewport_row: int, col: int) -> ChatPoint:
        lines = self._get_plain_lines()
        view_height = self._get_view_height()
        scroll_top = self._get_scroll_top(content_lines=len(lines), view_height=view_height)
        row = min(max(scroll_top + viewport_row, 0), max(0, len(lines) - 1))
        line = lines[row] if lines else ""
        return ChatPoint(row=row, col=_string_index_for_display_column(line, col))


def _apply_line_selection(
    line: list[tuple[str, str]],
    *,
    row: int,
    selection: tuple[ChatPoint, ChatPoint] | None,
) -> list[tuple[str, str]]:
    if selection is None:
        return line
    start, end = selection
    if row < start.row or row > end.row:
        return line
    line_text = "".join(text for _, text in line)
    line_length = len(line_text)
    left = start.col if row == start.row else 0
    right = end.col if row == end.row else line_length
    left = min(max(left, 0), line_length)
    right = min(max(right, 0), line_length)
    if left == right:
        return line
    result: list[tuple[str, str]] = []
    offset = 0
    for style, text in line:
        next_offset = offset + len(text)
        if next_offset <= left or offset >= right:
            result.append((style, text))
        else:
            before = max(0, left - offset)
            after = max(0, next_offset - right)
            middle_end = len(text) - after
            if before:
                result.append((style, text[:before]))
            if middle_end > before:
                selected_style = f"{style} class:selected" if style else "class:selected"
                result.append((selected_style, text[before:middle_end]))
            if after:
                result.append((style, text[middle_end:]))
        offset = next_offset
    return result


def _string_index_for_display_column(text: str, column: int) -> int:
    if column <= 0:
        return 0
    width = 0
    for index, char in enumerate(text):
        char_width = _char_display_width(char)
        if width + char_width > column:
            return index
        width += char_width
        if width == column:
            return index + 1
    return len(text)
