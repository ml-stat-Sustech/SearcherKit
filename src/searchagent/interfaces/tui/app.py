from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from searchagent.runtime.interactive_selection import infer_active_source
from searchagent.interfaces.tui.app_builder import TuiApplicationBuilder
from searchagent.interfaces.tui.chat.chat_history import ChatHistory
from searchagent.interfaces.tui.chat.chat_renderer import ChatRenderer
from searchagent.interfaces.tui.runtime.clipboard import _copy_text_to_clipboard
from searchagent.interfaces.tui.runtime.query_controller import QueryController
from searchagent.interfaces.tui.selection.active_selector import ModelSelector, SourceSelector
from searchagent.runtime.interactive_selection import ModelOption, active_model_label
from searchagent.runtime.interactive_selection import SelectionState
from searchagent.interfaces.tui.slash.slash_command import SlashCommandMenuState
from searchagent.interfaces.tui.slash.slash_command_handler import SlashCommandHandler
from searchagent.interfaces.tui.slash.slash_menu import SlashMenu, SlashMenuRenderer
from searchagent.interfaces.tui.ui.formatting import _formatted_lines
from searchagent.interfaces.tui.ui.input_field import InputField
from searchagent.interfaces.tui.ui.layout_geometry import FallbackSize, LayoutGeometry, TerminalSize
from searchagent.interfaces.tui.ui.selection_manager import SelectionManager, _apply_line_selection
from searchagent.interfaces.tui.ui.status_bar import build_status_bar
from searchagent.interfaces.tui.ui.view_state import TuiViewState
from searchagent.runtime.interactive import InteractiveQueryConfig


@dataclass(slots=True)
class _ChatLayout:
    key: tuple[int, int, bool, bool]
    parts: list[tuple[str, str]]
    lines: list[list[tuple[str, str]]]
    plain_lines: list[str]


class SearchAgentTui:
    """Thin prompt-toolkit shell for the SearchAgent interactive TUI."""

    def __init__(
        self,
        *,
        config: InteractiveQueryConfig,
        model_options: Sequence[ModelOption] | None = None,
        model_discovery_message: str = "",
    ) -> None:
        self.config = config
        self.session_state = SelectionState(
            active_source=infer_active_source(config.agent)
        )
        self.view_state = TuiViewState()
        self.chat_history = ChatHistory()
        self.renderer = ChatRenderer()
        self._chat_layout_cache: _ChatLayout | None = None
        self._model_options = list(model_options or [])
        self._model_discovery_message = model_discovery_message

        self.layout_geometry = LayoutGeometry(self._get_terminal_size)
        self.slash_menu_renderer = SlashMenuRenderer()

        self.query_controller = QueryController(
            config=config,
            session_state=self.session_state,
            chat_history=self.chat_history,
            view_state=self.view_state,
            on_refresh_needed=self.refresh,
        )
        self.slash_handler = SlashCommandHandler(
            config=config,
            selectors=self._build_selectors(model_discovery_message),
            session_state=self.session_state,
            view_state=self.view_state,
            chat_history=self.chat_history,
            is_running=self.query_controller.is_running,
            on_exit=self.exit_app,
            on_refresh_needed=self.refresh,
        )

        self._slash_menu_state = SlashCommandMenuState(self.slash_handler.commands())
        self.input_field = InputField(None)
        self.slash_menu = SlashMenu(self._slash_menu_state, self.input_field)

        self.chat_selection = SelectionManager(
            self.view_state,
            get_plain_lines=self._render_chat_plain_lines,
            get_view_height=self.chat_view_height,
            get_scroll_top=self._current_chat_scroll_top,
        )

        self._pt_app: Any = None
        self._set_intro()

    def _build_selectors(self, model_discovery_message: str) -> list[ModelSelector | SourceSelector]:
        selectors: list[ModelSelector | SourceSelector] = []
        if self._model_options:
            selectors.append(ModelSelector(self._model_options, model_discovery_message))
        if self.config.agent.sources:
            selectors.append(SourceSelector(self.config.agent.sources))
        return selectors

    def _get_terminal_size(self) -> TerminalSize:
        if self._pt_app is None:
            return FallbackSize()
        try:
            return self._pt_app.output.get_size()
        except (AttributeError, OSError, ValueError):
            return FallbackSize()

    def run(self, *, input: Any | None = None, output: Any | None = None) -> None:
        self._pt_app = TuiApplicationBuilder(
            shell=self,
            input_field=self.input_field,
            slash_menu=self.slash_menu,
            chat_selection=self.chat_selection,
            view_state=self.view_state,
            input=input,
            output=output,
        ).build()
        self._set_intro()
        try:
            self._pt_app.run()
        finally:
            self.query_controller.cancel()

    def exit_app(self) -> None:
        if self._pt_app is not None:
            self._pt_app.exit()

    def refresh(self) -> None:
        if self._pt_app is not None:
            self._pt_app.invalidate()

    # --- key-binding handlers ------------------------------------------------

    def handle_enter(self, event: Any) -> None:
        decision = self.slash_menu.decide_submit()
        if decision.action == "empty":
            return
        if decision.action in {"unknown", "argument_error", "invalid"}:
            self._show_tui_command_error(decision.message)
            self.refresh()
            return
        if decision.action == "submenu":
            self.slash_menu.open_submenu(decision.value)
            self.refresh()
            return
        if decision.action == "execute":
            self.execute_tui_command(decision.value)
            self.refresh()
            return
        if self.query_controller.is_running():
            return
        query_text = decision.value.strip()
        if query_text and (
            not self.view_state.query_history or self.view_state.query_history[-1] != query_text
        ):
            self.view_state.query_history.append(query_text)
        self.view_state.history_index = None
        self._clear_input_and_menu()
        event.app.create_background_task(self.query_controller.run_query(decision.value))

    def handle_up(self, event: Any) -> bool:
        if self.view_state.history_index is None:
            self.view_state.saved_input = self.input_field.text()
            self.view_state.history_index = len(self.view_state.query_history) - 1
        elif self.view_state.history_index > 0:
            self.view_state.history_index -= 1
        else:
            return False
        self.input_field.set_text(
            self.view_state.query_history[self.view_state.history_index], cursor_end=True
        )
        self.refresh()
        return True

    def handle_down(self, event: Any) -> bool:
        if self.view_state.history_index is None:
            return False
        if self.view_state.history_index < len(self.view_state.query_history) - 1:
            self.view_state.history_index += 1
            self.input_field.set_text(
                self.view_state.query_history[self.view_state.history_index], cursor_end=True
            )
        else:
            self.view_state.history_index = None
            self.input_field.set_text(self.view_state.saved_input, cursor_end=True)
        self.refresh()
        return True

    def handle_page_up(self, event: Any) -> None:
        self._scroll_chat(-max(1, self.chat_view_height() - 1))
        self.refresh()

    def handle_page_down(self, event: Any) -> None:
        self._scroll_chat(max(1, self.chat_view_height() - 1))
        self.refresh()

    def handle_ctrl_c(self, event: Any) -> None:
        if self.chat_selection.text():
            self._copy_chat_selection_to_clipboard()
            self.chat_selection.clear()
            self.refresh()
            return
        if self.query_controller.is_running():
            self.query_controller.cancel()
            self.refresh()
            return
        if self.input_field.text() or self.slash_menu.active_submenu is not None:
            self._clear_input_and_menu()
            self.refresh()
            return
        event.app.exit()

    def scroll_chat(self, delta: int) -> None:
        self._scroll_chat(delta)

    def execute_tui_command(self, command: str) -> bool:
        message = self.slash_handler.handle(command)
        if message is None:
            self._clear_input_and_menu()
            return True
        self._show_tui_command_error(message)
        return False

    def _show_tui_command_error(self, message: str) -> None:
        self.chat_history.append_tui_command_error(message or "Invalid TUI command.")
        self.view_state.chat_scroll_top = None

    def _clear_input_and_menu(self) -> None:
        self.slash_menu.close_submenu()
        self.input_field.set_text("")

    def _set_intro(self) -> None:
        self.view_state.chat_scroll_top = None
        self.chat_history.set_intro(
            model_label=active_model_label(self.config.agent.llm_client, self.session_state.active_model),
            has_model_menu=bool(self._model_options),
            discovery_message=self._model_discovery_message,
        )

    # --- rendering -----------------------------------------------------------

    def _render_chat_full(self) -> list[tuple[str, str]]:
        return self._chat_layout().parts

    def _render_chat_plain_lines(self) -> list[str]:
        return self._chat_layout().plain_lines

    def _chat_layout(self) -> _ChatLayout:
        chat_width = self.layout_geometry.chat_view_width()
        key = (
            self.chat_history.revision,
            chat_width,
            self.view_state.show_thinking,
            self.view_state.show_tool_detail,
        )
        cached = self._chat_layout_cache
        if cached is not None and cached.key == key:
            return cached
        parts = self.renderer.render_full(
            self.chat_history.entries(),
            chat_width=chat_width,
            show_thinking=self.view_state.show_thinking,
            show_tool_detail=self.view_state.show_tool_detail,
        )
        lines = _formatted_lines(parts)
        layout = _ChatLayout(
            key=key,
            parts=parts,
            lines=lines,
            plain_lines=["".join(text for _, text in line) for line in lines],
        )
        self._chat_layout_cache = layout
        return layout

    def render_chat_viewport(self) -> list[tuple[str, str]]:
        lines = self._chat_layout().lines
        view_height = self.chat_view_height()
        scroll_top = self._current_chat_scroll_top(
            content_lines=len(lines),
            view_height=view_height,
        )
        viewport = lines[scroll_top : scroll_top + view_height]
        result: list[tuple[str, str]] = []
        for offset, line in enumerate(viewport):
            result.extend(
                _apply_line_selection(
                    line,
                    row=scroll_top + offset,
                    selection=self.chat_selection.bounds(),
                )
            )
            result.append(("", "\n"))
        for _ in range(max(0, view_height - len(viewport))):
            result.append(("", "\n"))
        return result

    def render_scrollbar(self) -> list[tuple[str, str]]:
        view_height = self.chat_view_height()
        content_lines = len(self._chat_layout().lines)
        if content_lines <= view_height:
            return [("class:separator", " \n" * max(1, view_height))]
        thumb_start, thumb_size = self.layout_geometry.scrollbar_thumb(
            stored_scroll_top=self.view_state.chat_scroll_top,
            content_lines=content_lines,
            view_height=view_height,
        )
        parts: list[tuple[str, str]] = []
        for index in range(view_height):
            char = "#" if thumb_start <= index < thumb_start + thumb_size else "|"
            parts.append(
                ("class:scrollbar-thumb" if char == "#" else "class:scrollbar-track", f"{char}\n")
            )
        return parts

    def render_status(self) -> list[tuple[str, str]]:
        content = build_status_bar(self.config, self.session_state)
        return [
            ("class:status-label", "Model:"),
            ("class:status-model", content.model_label),
            ("class:status", " · "),
            ("class:status-label", "Source:"),
            ("class:status-source", content.source_label),
            ("class:status", " · "),
            ("class:status-label", "Records:"),
            ("class:status-records", content.record_label),
        ]

    def render_kicker(self) -> list[tuple[str, str]]:
        """One-line live progress label (blank when idle)."""
        if not self.query_controller.is_running():
            return [("class:running-kicker", "")]
        return [("class:running-kicker", f" {self._spinner_marker()} searching...\n")]

    def render_slash_candidates(self) -> list[tuple[str, str]]:
        return self.slash_menu_renderer.render(self.slash_menu.menu_state)

    def _spinner_marker(self) -> str:
        frames = ("[|]", "[/]", "[-]", "[\\]")
        return frames[self.view_state.spinner_frame % len(frames)]

    def copy_chat_selection_to_clipboard(self) -> None:
        text = self.chat_selection.text()
        if not text:
            return
        _copy_text_to_clipboard(text)

    # --- layout helpers exposed to the builder -------------------------------

    def input_view_height(self) -> int:
        return self.layout_geometry.input_view_height(
            self.input_field.text(),
            slash_visible=self.slash_candidates_visible(),
        )

    def slash_candidates_height(self) -> int:
        return self.slash_menu.candidates_height()

    def chat_view_height(self) -> int:
        return self.layout_geometry.chat_view_height(
            input_height=self.input_view_height(),
            slash_visible=self.slash_candidates_visible(),
        )

    def is_running(self) -> bool:
        return self.query_controller.is_running()

    def slash_candidates_visible(self) -> bool:
        return self.slash_menu.is_active()

    def _current_chat_scroll_top(self, *, content_lines: int, view_height: int) -> int:
        return self.layout_geometry.current_chat_scroll_top(
            stored_scroll_top=self.view_state.chat_scroll_top,
            content_lines=content_lines,
            view_height=view_height,
        )

    def _scroll_chat(self, delta: int) -> None:
        lines = self._chat_layout().lines
        view_height = self.chat_view_height()
        content_lines = len(lines)
        self.view_state.chat_scroll_top = self.layout_geometry.next_scroll_top(
            stored_scroll_top=self.view_state.chat_scroll_top,
            content_lines=content_lines,
            view_height=view_height,
            delta=delta,
        )
