from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from searchagent.interfaces.tui.chat.chat_history import ChatHistory
from searchagent.interfaces.tui.selection.active_selector import ModelSelector, SourceSelector
from searchagent.runtime.interactive_selection import SelectionState
from searchagent.interfaces.tui.slash.slash_command import TuiCommand
from searchagent.interfaces.tui.ui.view_state import TuiViewState
from searchagent.runtime.interactive import InteractiveQueryConfig


class SlashCommandHandler:
    """Dispatches all slash commands and builds the slash menu."""

    def __init__(
        self,
        *,
        config: InteractiveQueryConfig,
        selectors: Sequence[ModelSelector | SourceSelector],
        session_state: SelectionState,
        view_state: TuiViewState,
        chat_history: ChatHistory,
        is_running: Callable[[], bool],
        on_exit: Callable[[], None],
        on_refresh_needed: Callable[[], None],
    ) -> None:
        self._config = config
        self._selectors = list(selectors)
        self._session_state = session_state
        self._view_state = view_state
        self._chat_history = chat_history
        self._is_running = is_running
        self._on_exit = on_exit
        self._on_refresh_needed = on_refresh_needed

    def commands(self) -> list[TuiCommand]:
        """Return the full list of TuiCommand entries for the slash menu."""
        commands: list[TuiCommand] = []
        for selector in self._selectors:
            items = selector.list_items()
            if items:
                commands.append(
                    TuiCommand(selector.prefix, f"Choose the Active {selector.prefix.capitalize()}", kind="submenu")
                )
                for item in items:
                    commands.append(
                        TuiCommand(
                            item.command_name,
                            f"Use {selector.prefix[:-1]} {item.label}",
                            parent=selector.prefix,
                        )
                    )
        commands.extend(_builtin_commands())
        return commands

    def handle(self, command: str) -> str | None:
        """Execute a slash command (without the leading ``/``).

        Returns an error message, or ``None`` when the command succeeded.
        """
        if command == "thinking":
            self._view_state.show_thinking = not self._view_state.show_thinking
            self._on_refresh_needed()
            return None
        if command == "tool-detail":
            self._view_state.show_tool_detail = not self._view_state.show_tool_detail
            self._on_refresh_needed()
            return None
        if command == "clear":
            return self._clear()
        if command == "quit":
            return self._quit()

        for selector in self._selectors:
            option = selector.parse_command(command)
            if option is not None:
                return self._apply_selection(selector, option)

        return f"Unknown command: /{command}"

    def _clear(self) -> str | None:
        if self._is_running():
            return "Cannot clear while a query is running. Press Ctrl+C to cancel first."
        self._chat_history.set_intro(
            model_label=self._model_label(),
            has_model_menu=self._has_model_menu(),
            discovery_message=self._discovery_message(),
        )
        self._chat_history.append(
            role="meta",
            title="Clear",
            body="Conversation cleared. Ready for new queries.",
            style="class:meta",
        )
        self._view_state.chat_scroll_top = None
        self._on_refresh_needed()
        return None

    def _quit(self) -> str | None:
        if self._is_running():
            return "Cannot quit while a query is running. Press Ctrl+C to cancel first."
        self._on_exit()
        return None

    def _apply_selection(self, selector: ModelSelector | SourceSelector, option: Any) -> str | None:
        if self._is_running():
            return (
                f"Cannot switch {selector.display_name.lower()} while a query is running. "
                "Press Ctrl+C to cancel first."
            )
        label = selector.apply(self._session_state, option)
        self._chat_history.append_selection_entry(selector.display_name, label)
        self._view_state.chat_scroll_top = None
        self._on_refresh_needed()
        return None

    def _model_label(self) -> str:
        active_model = self._session_state.active_model
        if active_model is not None:
            return active_model.label
        client = self._config.agent.llm_client
        provider = str(client.type or "provider")
        model = str(client.model or "model")
        return f"{provider}/{model}"

    def _has_model_menu(self) -> bool:
        for selector in self._selectors:
            if isinstance(selector, ModelSelector) and selector.list_items():
                return True
        return False

    def _discovery_message(self) -> str:
        for selector in self._selectors:
            if isinstance(selector, ModelSelector):
                return selector.discovery_message
        return ""


def _builtin_commands() -> list[TuiCommand]:
    return [
        TuiCommand("thinking", "Toggle assistant reasoning visibility"),
        TuiCommand("tool-detail", "Toggle tool call detail visibility"),
        TuiCommand("clear", "Clear current live view"),
        TuiCommand("quit", "Quit the TUI"),
    ]
