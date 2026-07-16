from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True, slots=True)
class TuiCommand:
    name: str
    description: str
    kind: Literal["command", "submenu"] = "command"
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class SlashSubmitDecision:
    action: Literal["empty", "query", "execute", "submenu", "unknown", "argument_error", "invalid"]
    value: str = ""
    message: str = ""


class SlashCommandMenuState:
    def __init__(self, commands: Sequence[TuiCommand]) -> None:
        self.commands = list(commands)
        self.candidates: list[TuiCommand] = []
        self.selected_index = 0
        self.prefix = ""
        self.active_submenu: str | None = None
        self.dismissed_text: str | None = None

    @property
    def is_active(self) -> bool:
        return bool(self.candidates)

    @property
    def selected_command(self) -> TuiCommand | None:
        if not self.candidates:
            return None
        return self.candidates[min(self.selected_index, len(self.candidates) - 1)]

    def update(self, text: str) -> None:
        if self.dismissed_text == text:
            self.candidates = []
            self.selected_index = 0
            return
        if self.dismissed_text is not None and self.dismissed_text != text:
            self.dismissed_text = None
        previous = self.selected_command.name if self.selected_command is not None else None
        if self.active_submenu is not None and not text.startswith("/"):
            self.prefix = text.strip()
            self.candidates = self._active_submenu_candidates(self.prefix)
        else:
            if text.startswith("/"):
                self.active_submenu = None
            if not re.fullmatch(r"/\S*", text):
                self.prefix = ""
                self.candidates = []
                self.selected_index = 0
                return
            self.prefix = text[1:]
            self.candidates = self._candidates_for_prefix(self.prefix)
        if not self.candidates:
            self.selected_index = 0
            return
        if previous is not None:
            for index, command in enumerate(self.candidates):
                if command.name == previous:
                    self.selected_index = index
                    return
        self.selected_index = 0

    def open_submenu(self, name: str) -> None:
        self.active_submenu = name
        self.prefix = ""
        self.candidates = self._active_submenu_candidates("")
        self.selected_index = 0

    def close_submenu(self) -> None:
        self.active_submenu = None

    def dismiss(self, text: str) -> None:
        self.dismissed_text = text
        self.active_submenu = None
        self.candidates = []
        self.selected_index = 0

    def _active_submenu_candidates(self, filter_text: str) -> list[TuiCommand]:
        if self.active_submenu is None:
            return []
        children = [command for command in self.commands if command.parent == self.active_submenu]
        if not filter_text:
            return children
        effective = f"{self.active_submenu}:{filter_text}"
        return [command for command in children if command.name.startswith(effective)]

    def _candidates_for_prefix(self, prefix: str) -> list[TuiCommand]:
        submenu = self._submenu_for_prefix(prefix)
        if submenu is not None:
            children = [command for command in self.commands if command.parent == submenu]
            if prefix == submenu or prefix == f"{submenu}:":
                return children
            if prefix.startswith(f"{submenu}:"):
                return [c for c in children if c.name.startswith(prefix)]
            # No colon: e.g. "modelso" → search children with effective prefix "models:o"
            remainder = prefix[len(submenu):]
            effective = f"{submenu}:{remainder}"
            return [c for c in children if c.name.startswith(effective)]
        return [command for command in self.commands if command.parent is None and command.name.startswith(prefix)]

    def _submenu_for_prefix(self, prefix: str) -> str | None:
        for command in self.commands:
            if command.kind != "submenu" or command.parent is not None:
                continue
            if prefix == command.name or prefix.startswith(f"{command.name}:"):
                return command.name
            if prefix.startswith(command.name):
                return command.name
        return None

    def move_selection(self, delta: int) -> None:
        if not self.candidates:
            return
        self.selected_index = (self.selected_index + delta) % len(self.candidates)

    def decide_submit(self, raw_text: str) -> SlashSubmitDecision:
        if self.active_submenu is not None and not raw_text.startswith("/"):
            selected = self.selected_command
            if selected is None:
                return SlashSubmitDecision("empty")
            return SlashSubmitDecision("execute", value=selected.name)
        if not raw_text.strip():
            return SlashSubmitDecision("empty")
        if not raw_text.startswith("/"):
            return SlashSubmitDecision("query", raw_text.strip())
        text = raw_text.strip()
        try:
            parts = shlex.split(text[1:])
        except ValueError as exc:
            return SlashSubmitDecision("invalid", message=f"Invalid slash command: {exc}")
        if not parts:
            selected = self.selected_command
            if selected is None:
                return SlashSubmitDecision("unknown", value="/")
            if selected.kind == "submenu":
                return SlashSubmitDecision("submenu", value=selected.name)
            return SlashSubmitDecision("execute", value=selected.name)
        name, *args = parts
        if args:
            return SlashSubmitDecision("argument_error", value=name, message=f"/{name} does not accept arguments")
        command = self._find_command(name)
        if command is not None and command.kind == "submenu":
            return SlashSubmitDecision("submenu", value=command.name)
        selected = self.selected_command
        if selected is not None:
            if selected.kind == "submenu":
                return SlashSubmitDecision("submenu", value=selected.name)
            return SlashSubmitDecision("execute", value=selected.name)
        if command is None:
            return SlashSubmitDecision("unknown", value=name, message=f"Unknown command: /{name}")
        if command.kind == "submenu":
            return SlashSubmitDecision("submenu", value=command.name)
        return SlashSubmitDecision("execute", value=command.name)

    def _find_command(self, name: str) -> TuiCommand | None:
        for command in self.commands:
            if command.name == name:
                return command
        return None
