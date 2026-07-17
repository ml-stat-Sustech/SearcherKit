"""Terminal-size-driven layout geometry for the TUI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class TerminalSize(Protocol):
    """Anything that exposes rows and columns, such as prompt-toolkit's Size."""

    rows: int
    columns: int


@dataclass(frozen=True)
class GeometryConfig:
    """Constants that parameterize the layout."""

    slash_menu_height: int = 5
    kicker_height: int = 1
    status_bar_height: int = 1
    input_separator_height: int = 2
    min_chat_height: int = 4
    max_input_height: int = 6
    prompt_width: int = 2


class LayoutGeometry:
    """Computes all terminal-dependent sizes and scroll positions."""

    def __init__(
        self,
        get_terminal_size: Callable[[], TerminalSize],
        *,
        config: GeometryConfig | None = None,
    ) -> None:
        self._get_terminal_size = get_terminal_size
        self._config = config or GeometryConfig()

    def terminal_rows(self) -> int:
        return self._safe_size().rows

    def chat_view_width(self) -> int:
        return max(20, self._safe_size().columns - 1)

    def input_view_width(self) -> int:
        return max(1, self._safe_size().columns)

    def chat_view_height(
        self,
        *,
        input_height: int,
        slash_visible: bool,
    ) -> int:
        reserved = (
            self._config.status_bar_height
            + self._config.input_separator_height
            + input_height
            + self._slash_height(slash_visible)
            + self._config.kicker_height
        )
        return max(self._config.min_chat_height, self.terminal_rows() - reserved)

    def input_view_height(self, text: str, *, slash_visible: bool) -> int:
        available = self._available_dynamic_height(slash_visible=slash_visible)
        return min(
            max(1, self._input_visual_line_count(text)),
            self._config.max_input_height,
            max(1, available),
        )

    def current_chat_scroll_top(
        self,
        *,
        stored_scroll_top: int | None,
        content_lines: int,
        view_height: int,
    ) -> int:
        max_scroll = max(0, content_lines - view_height)
        if stored_scroll_top is None:
            return max_scroll
        return min(max(stored_scroll_top, 0), max_scroll)

    def next_scroll_top(
        self,
        *,
        stored_scroll_top: int | None,
        content_lines: int,
        view_height: int,
        delta: int,
    ) -> int | None:
        """Return the new stored scroll top, or None when pinned to the bottom."""

        current = self.current_chat_scroll_top(
            stored_scroll_top=stored_scroll_top,
            content_lines=content_lines,
            view_height=view_height,
        )
        max_scroll = max(0, content_lines - view_height)
        next_top = min(max(current + delta, 0), max_scroll)
        return None if next_top >= max_scroll else next_top

    def scrollbar_thumb(
        self,
        *,
        stored_scroll_top: int | None,
        content_lines: int,
        view_height: int,
    ) -> tuple[int, int]:
        """Return (thumb_start, thumb_size) for the vertical scrollbar."""

        if content_lines <= view_height:
            return 0, view_height
        thumb_size = max(1, round((view_height / content_lines) * view_height))
        max_scroll = max(1, content_lines - view_height)
        scroll_top = self.current_chat_scroll_top(
            stored_scroll_top=stored_scroll_top,
            content_lines=content_lines,
            view_height=view_height,
        )
        thumb_start = round((scroll_top / max_scroll) * (view_height - thumb_size))
        return thumb_start, thumb_size

    def _available_dynamic_height(self, *, slash_visible: bool) -> int:
        return max(
            1,
            self.terminal_rows()
            - self._config.status_bar_height
            - self._config.input_separator_height
            - self._config.kicker_height
            - self._slash_height(slash_visible)
            - self._config.min_chat_height,
        )

    def _input_visual_line_count(self, text: str) -> int:
        width = max(1, self.input_view_width())
        total = 0
        for index, line in enumerate(text.split("\n")):
            available_width = max(1, width - (self._config.prompt_width if index == 0 else 0))
            total += max(1, (len(line) + available_width - 1) // available_width)
        return total

    def _slash_height(self, visible: bool) -> int:
        return self._config.slash_menu_height if visible else 0

    def _safe_size(self) -> TerminalSize:
        try:
            size = self._get_terminal_size()
        except (AttributeError, OSError, ValueError):
            return FallbackSize()
        return size


class FallbackSize:
    rows = 20
    columns = 80
