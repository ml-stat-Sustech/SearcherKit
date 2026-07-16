"""Slash menu rendering and navigation for the TUI."""

from __future__ import annotations

from searcherkit.interfaces.tui.slash.slash_command import SlashCommandMenuState, SlashSubmitDecision
from searcherkit.interfaces.tui.ui.input_field import InputField


class SlashMenu:
    """Coordinates input text and slash-menu state for keyboard interaction."""

    def __init__(
        self,
        menu_state: SlashCommandMenuState,
        input_field: InputField,
        *,
        candidates_height: int = 5,
    ) -> None:
        self._menu_state = menu_state
        self._input = input_field
        self._candidates_height = candidates_height

    @property
    def menu_state(self) -> SlashCommandMenuState:
        return self._menu_state

    @menu_state.setter
    def menu_state(self, value: SlashCommandMenuState) -> None:
        self._menu_state = value

    def is_active(self) -> bool:
        self._sync()
        return self._menu_state.is_active

    @property
    def active_submenu(self) -> str | None:
        return self._menu_state.active_submenu

    def move_selection(self, delta: int) -> bool:
        self._sync()
        if not self._menu_state.is_active:
            return False
        self._menu_state.move_selection(delta)
        return True

    def complete_selected(self) -> bool:
        self._sync()
        selected = self._menu_state.selected_command
        if selected is None:
            return False
        suffix = ":" if selected.kind == "submenu" else ""
        self._input.set_text(f"/{selected.name}{suffix}")
        return True

    def dismiss(self) -> bool:
        self._sync()
        if not self._menu_state.is_active and self._menu_state.active_submenu is None:
            return False
        self._menu_state.dismiss(self._input.text())
        return True

    def decide_submit(self) -> SlashSubmitDecision:
        self._sync()
        return self._menu_state.decide_submit(self._input.text())

    def open_submenu(self, name: str) -> None:
        self._menu_state.open_submenu(name)
        self._input.set_text("")

    def close_submenu(self) -> None:
        self._menu_state.close_submenu()

    def candidates_height(self) -> int:
        return self._candidates_height if self.is_active() else 0

    def _sync(self) -> None:
        self._menu_state.update(self._input.text())


class SlashMenuRenderer:
    """Renders slash-menu state into prompt-toolkit formatted text tuples."""

    def __init__(self, visible_count: int = 5) -> None:
        self.visible_count = max(1, visible_count)

    def render(self, menu_state: SlashCommandMenuState) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        start = 0
        if menu_state.selected_index >= self.visible_count:
            start = menu_state.selected_index - self.visible_count + 1
        for offset, command in enumerate(menu_state.candidates[start : start + self.visible_count]):
            index = start + offset
            selected = index == menu_state.selected_index
            style = "class:slash-selected" if selected else "class:slash-menu"
            prefix = "›" if selected else " "
            suffix = " ›" if command.kind == "submenu" else ""
            parts.append((style, f"{prefix} /{command.name}{suffix} {command.description}\n"))
        return parts or [("class:slash-menu", "")]
