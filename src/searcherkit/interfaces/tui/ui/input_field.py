"""Input field controller for the TUI."""

from __future__ import annotations

from typing import Any


class InputField:
    """Wraps the prompt-toolkit input field so callers avoid buffer internals."""

    def __init__(self, input_field: Any) -> None:
        self._input_field = input_field

    def text(self) -> str:
        if self._input_field is None:
            return ""
        return self._input_field.text

    def set_text(self, text: str, *, cursor_end: bool = False) -> None:
        if self._input_field is None:
            return
        self._input_field.text = text
        if cursor_end:
            buffer = getattr(self._input_field, "buffer", None)
            if buffer is not None:
                buffer.cursor_position = len(text)

    def insert_text(self, text: str) -> None:
        """Insert text at the current cursor position."""

        buffer = getattr(self._input_field, "buffer", None)
        if buffer is not None and hasattr(buffer, "insert_text"):
            buffer.insert_text(text)

    def move_cursor_up(self) -> None:
        """Move the input cursor up one visual line."""

        self._call_buffer_command("cursor_up")

    def move_cursor_down(self) -> None:
        """Move the input cursor down one visual line."""

        self._call_buffer_command("cursor_down")

    def _call_buffer_command(self, command: str) -> None:
        buffer = getattr(self._input_field, "buffer", None)
        if buffer is None or not hasattr(buffer, command):
            return
        getattr(buffer, command)()

    def set_input_field(self, input_field: Any) -> None:
        """Bind a concrete prompt-toolkit input field after it is constructed."""

        self._input_field = input_field
