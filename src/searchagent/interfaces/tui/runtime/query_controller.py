from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from typing import Any

from searchagent.runtime.interactive_selection import apply_active_source
from searchagent.interfaces.tui.chat.chat_history import ChatHistory
from searchagent.interfaces.tui.runtime.provider_errors import (
    _friendly_searchagent_error_message,
    _openai_provider_error_types,
    friendly_provider_error_message,
)
from searchagent.runtime.interactive_selection import active_model_label, apply_active_model
from searchagent.runtime.interactive_selection import SelectionState
from searchagent.interfaces.tui.ui.view_state import SPINNER_FRAMES, TuiViewState
from searchagent.common.errors import SearchAgentError
from searchagent.runtime.interactive import InteractiveQueryConfig, InteractiveQueryRunner
from searchagent.common.live_events import LiveEvent


class QueryController:
    """Manages the lifecycle of one Interactive Query Run."""

    def __init__(
        self,
        *,
        config: InteractiveQueryConfig,
        session_state: SelectionState,
        chat_history: ChatHistory,
        view_state: TuiViewState,
        on_refresh_needed: Callable[[], None],
    ) -> None:
        self._config = config
        self._session_state = session_state
        self._chat_history = chat_history
        self._view_state = view_state
        self._on_refresh_needed = on_refresh_needed
        self.running = False
        self._current_task: asyncio.Task[Any] | None = None
        self._spinner_task: asyncio.Task[Any] | None = None

    def is_running(self) -> bool:
        if self.running:
            return True
        return self._current_task is not None and not self._current_task.done()

    def cancel(self) -> None:
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    async def run_query(self, query: str) -> None:
        if not query.strip():
            return
        self.running = True
        self._chat_history.clear()
        self._start_spinner()
        self._on_refresh_needed()

        async def on_event(event: LiveEvent) -> None:
            self._chat_history.append_event(event)
            self._update_spinner()
            self._on_refresh_needed()

        run_config = self.build_run_config()
        self._current_task = asyncio.create_task(
            InteractiveQueryRunner(config=run_config).run_query(query, live_event_sink=on_event)
        )
        try:
            await self._current_task
        except asyncio.CancelledError:
            self._chat_history.append_cancelled()
            raise
        except (SearchAgentError, OSError, TimeoutError) as exc:
            self._chat_history.append_run_error(
                "Run Error", _friendly_searchagent_error_message(exc)
            )
        except _openai_provider_error_types() as exc:
            self._chat_history.append_run_error(
                "Provider Error",
                friendly_provider_error_message(
                    exc, model_label=active_model_label(self._config.agent.llm_client, self._session_state.active_model)
                ),
            )
        finally:
            self.running = False
            self._current_task = None
            self._stop_spinner()
            self._on_refresh_needed()

    def build_run_config(self) -> InteractiveQueryConfig:
        run_config = copy.deepcopy(self._config)
        active_model = self._session_state.active_model
        if active_model is not None:
            apply_active_model(run_config.agent.llm_client, active_model)
        active_source = self._session_state.active_source
        if active_source is not None and active_source.name is not None:
            apply_active_source(run_config.agent, active_source.name)
        return run_config

    def _start_spinner(self) -> None:
        if self._spinner_task is not None and not self._spinner_task.done():
            return
        if not self._should_spin():
            return
        self._spinner_task = asyncio.create_task(self._spinner_loop())

    def _stop_spinner(self) -> None:
        if self._spinner_task is not None and not self._spinner_task.done():
            self._spinner_task.cancel()
        self._spinner_task = None

    def _update_spinner(self) -> None:
        if self._should_spin():
            self._start_spinner()
        else:
            self._stop_spinner()

    def _should_spin(self) -> bool:
        return self.running or self._chat_history.has_unfinished_thinking_or_tool()

    async def _spinner_loop(self) -> None:
        try:
            while self._should_spin():
                await asyncio.sleep(0.12)
                self._view_state.spinner_frame = (
                    getattr(self._view_state, "spinner_frame", 0) + 1
                ) % len(SPINNER_FRAMES)
                self._on_refresh_needed()
        except asyncio.CancelledError:
            return
