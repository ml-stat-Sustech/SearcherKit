"""Prompt-toolkit application assembly for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from searcherkit.interfaces.tui.slash.slash_menu import SlashMenu
from searcherkit.interfaces.tui.ui.input_field import InputField
from searcherkit.interfaces.tui.ui.selection_manager import SelectionManager
from searcherkit.interfaces.tui.ui.view_state import TuiViewState

if TYPE_CHECKING:
    from searcherkit.interfaces.tui.app import SearcherKitTui


_PromptToolkitEscapeTimeoutSeconds = 0.03


class TuiApplicationBuilder:
    """Builds the prompt-toolkit Application from collaborators."""

    def __init__(
        self,
        shell: SearcherKitTui,
        *,
        input_field: InputField,
        slash_menu: SlashMenu,
        chat_selection: SelectionManager,
        view_state: TuiViewState,
        input: Any | None = None,
        output: Any | None = None,
    ) -> None:
        self._shell = shell
        self._input_field = input_field
        self._slash_menu = slash_menu
        self._chat_selection = chat_selection
        self._view_state = view_state
        self._input = input
        self._output = output

    def build(self) -> Any:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.filters import Condition
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.mouse_events import MouseEventType
            from prompt_toolkit.widgets import TextArea
        except ImportError as exc:
            raise ImportError(
                "searcher tui requires prompt-toolkit. Install it with "
                "`pip install prompt-toolkit` or add the project tui extra."
            ) from exc

        input_field = self._input_field
        slash_menu = self._slash_menu

        bindings = KeyBindings()

        @bindings.add("enter", eager=True)
        def _enter(event: Any) -> None:
            self._shell.handle_enter(event)

        @bindings.add("tab", eager=True)
        def _tab(event: Any) -> None:
            if slash_menu.complete_selected():
                self._shell.refresh()
                return
            input_field.insert_text("\t")

        @bindings.add("up", eager=True)
        def _up(event: Any) -> None:
            if slash_menu.move_selection(-1):
                self._shell.refresh()
                return
            if slash_menu.is_active():
                input_field.move_cursor_up()
                return
            if self._view_state.query_history:
                self._shell.handle_up(event)
                return
            input_field.move_cursor_up()

        @bindings.add("down", eager=True)
        def _down(event: Any) -> None:
            if slash_menu.move_selection(1):
                self._shell.refresh()
                return
            if slash_menu.is_active():
                input_field.move_cursor_down()
                return
            if self._view_state.history_index is not None:
                self._shell.handle_down(event)
                return
            input_field.move_cursor_down()

        @bindings.add("c-j", eager=True)
        def _ctrl_j(event: Any) -> None:
            input_field.insert_text("\n")

        @bindings.add("escape", eager=True)
        def _escape(event: Any) -> None:
            if slash_menu.dismiss():
                self._shell.refresh()

        @bindings.add("pageup", eager=True)
        def _page_up(event: Any) -> None:
            self._shell.handle_page_up(event)

        @bindings.add("pagedown", eager=True)
        def _page_down(event: Any) -> None:
            self._shell.handle_page_down(event)

        @bindings.add("c-c", eager=True)
        def _cancel_clear_or_quit(event: Any) -> None:
            self._shell.handle_ctrl_c(event)

        @bindings.add("c-o", eager=True)
        def _toggle_detail(event: Any) -> None:
            self._shell.handle_ctrl_o(event)

        class _ChatControl(FormattedTextControl):
            def mouse_handler(_self, mouse_event):
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    self._chat_selection.clear()
                    self._shell.scroll_chat(-3)
                    self._shell.refresh()
                    return None
                if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    self._chat_selection.clear()
                    self._shell.scroll_chat(3)
                    self._shell.refresh()
                    return None
                if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
                    self._chat_selection.start(mouse_event.position.y, mouse_event.position.x)
                    self._shell.refresh()
                    return None
                if mouse_event.event_type == MouseEventType.MOUSE_MOVE:
                    self._chat_selection.update(mouse_event.position.y, mouse_event.position.x)
                    self._shell.refresh()
                    return None
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    has_text, _ = self._chat_selection.finish(
                        mouse_event.position.y, mouse_event.position.x
                    )
                    if has_text:
                        self._shell.copy_chat_selection_to_clipboard()
                    self._shell.refresh()
                    return None
                return NotImplemented

        chat_window = Window(_ChatControl(self._shell.render_chat_viewport), wrap_lines=True)
        scrollbar = ConditionalContainer(
            Window(FormattedTextControl(self._shell.render_scrollbar), width=1),
            filter=Condition(lambda: not self._shell.is_splash()),
        )
        text_area = TextArea(
            height=self._shell.input_view_height,
            prompt="> ",
            multiline=True,
            wrap_lines=True,
        )
        input_field.set_input_field(text_area)

        slash_candidates = ConditionalContainer(
            Window(
                FormattedTextControl(self._shell.render_slash_candidates),
                height=self._shell.slash_candidates_height,
            ),
            filter=Condition(slash_menu.is_active),
        )
        # Blank spacer keeps running kicker text one row below the chat content.
        # Splash page hides kicker chrome and shows a one-line input hint instead.
        kicker_spacer = ConditionalContainer(
            Window(height=1),
            filter=Condition(lambda: not self._shell.is_splash()),
        )
        kicker = ConditionalContainer(
            Window(FormattedTextControl(self._shell.render_kicker), height=1),
            filter=Condition(lambda: not self._shell.is_splash()),
        )
        splash_hint = ConditionalContainer(
            Window(FormattedTextControl(self._shell.render_splash_hint), height=1),
            filter=Condition(self._shell.is_splash),
        )
        status_window = ConditionalContainer(
            Window(FormattedTextControl(self._shell.render_status), height=1),
            filter=Condition(lambda: not self._shell.is_splash()),
        )
        root = HSplit([
            VSplit([chat_window, scrollbar]),
            kicker_spacer,
            kicker,
            splash_hint,
            Window(height=1, char="─"),
            text_area,
            Window(height=1, char="─"),
            slash_candidates,
            status_window,
        ])
        pt_app = Application(
            layout=Layout(root, focused_element=text_area),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            style=self._tui_style(),
            input=self._input,
            output=self._output,
        )
        pt_app.ttimeoutlen = _PromptToolkitEscapeTimeoutSeconds
        pt_app.timeoutlen = _PromptToolkitEscapeTimeoutSeconds
        return pt_app

    def _tui_style(self) -> Any:
        from searcherkit.interfaces.tui.ui.formatting import _tui_style
        return _tui_style()
