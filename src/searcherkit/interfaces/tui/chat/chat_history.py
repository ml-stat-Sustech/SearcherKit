from __future__ import annotations

from collections.abc import Sequence

from searcherkit.interfaces.tui.chat.conversation_entry import ConversationEntry
from searcherkit.interfaces.tui.ui.formatting import _extract_final_answer_body
from searcherkit.common.live_events import LiveEvent


class ChatHistory:
    """Owns the sequence of chat entries shown in the TUI.

    This is the only component that mutates the entry list. All other
    collaborators receive an immutable snapshot via ``entries()``.
    """

    def __init__(self) -> None:
        self._entries: list[ConversationEntry] = []
        self._revision = 0

    @property
    def revision(self) -> int:
        """Return a token that changes whenever visible history may have changed."""
        return self._revision

    def _touch(self) -> None:
        self._revision += 1

    def clear(self) -> None:
        self._entries = []
        self._touch()

    def set_intro(
        self,
        *,
        model_label: str,
        has_model_menu: bool,
        discovery_message: str = "",
    ) -> None:
        body = (
            f"Model: {model_label}\n"
            "Type a query and press Enter. Each query starts a fresh independent run."
        )
        if has_model_menu:
            body += "\nType /models to open the Active Model submenu for future runs."
        if discovery_message:
            body += f"\n{discovery_message}"
        self._entries = [
            ConversationEntry(
                role="intro",
                title="SearcherKit Interactive Query",
                body=body,
                style="class:meta",
            )
        ]
        self._touch()

    def append(
        self,
        *,
        role: str,
        title: str,
        body: str = "",
        style: str = "",
        status: str = "",
    ) -> None:
        self._entries.append(
            ConversationEntry(role=role, title=title, body=body, style=style, status=status)
        )
        self._touch()

    def append_selection_entry(self, prefix: str, label: str) -> None:
        title = f"Active {prefix.capitalize()}"
        entry = ConversationEntry(
            role="meta",
            title=title,
            body=label,
            style="class:meta",
        )
        if self._entries and self._entries[-1].role == "meta" and self._entries[-1].title == title:
            self._entries[-1] = entry
        else:
            self._entries.append(entry)
        self._touch()

    def append_run_error(self, title: str, body: str) -> None:
        self._entries.append(
            ConversationEntry(
                role="error",
                title=title,
                body=body,
                style="class:error",
                status="failed",
            )
        )
        self._touch()

    def append_tui_command_error(self, body: str) -> None:
        self._entries.append(
            ConversationEntry(
                role="error",
                title="TUI Command Error",
                body=body,
                style="class:error",
                status="failed",
            )
        )
        self._touch()

    def append_cancelled(self, message: str = "") -> None:
        self._entries.append(
            ConversationEntry(
                role="error",
                title="CANCELLED",
                body=message
                or "Interactive query run task was cancelled before a record could be returned.",
                style="class:error",
                status="failed",
            )
        )
        self._touch()

    def append_event(self, event: LiveEvent) -> None:
        kind = event.kind
        if kind in {"run_started", "run_completed", "assistant_turn_started"}:
            return
        self._touch()
        if kind == "run_cancelled":
            self.append_cancelled(event.message)
        elif kind == "run_failed":
            self.append_run_error("FAILED", event.message)
        elif kind == "user_message":
            self._entries.append(
                ConversationEntry(role="user", title="USER", body=event.message, style="class:user")
            )
        elif kind == "assistant_delta":
            data = event.data
            delta = str(data.get("delta") or event.message or "")
            if not delta:
                return
            turn_meta = f"turn {data.get('turn')}" if data.get("turn") else ""
            field = str(data.get("field") or "content")
            if field == "thinking":
                thinking_entry = self._find_running_thinking(turn_meta=turn_meta)
                if thinking_entry is None:
                    thinking_entry = ConversationEntry(
                        role="thinking",
                        title="thinking...",
                        meta=turn_meta,
                        style="class:thinking",
                        status="running",
                    )
                    self._entries.append(thinking_entry)
                thinking_entry.thinking += delta
            else:
                existing = self._find_streaming_assistant_entry(turn_meta=turn_meta, create=False)
                # Whitespace-only content is not a user-visible message; do not open a block.
                if existing is None and not delta.strip():
                    return
                assistant_entry = self._find_streaming_assistant_entry(turn_meta=turn_meta)
                assert assistant_entry is not None
                assistant_entry.body += delta
        elif kind == "assistant_message":
            data = event.data
            thinking = data.get("thinking")
            content = data.get("content")
            tool_calls = data.get("tool_calls") or []
            turn_meta = f"turn {data.get('turn')}" if data.get("turn") else ""
            self._complete_latest_thinking(
                turn_meta=turn_meta, reasoning=str(thinking) if thinking else ""
            )
            # Message = non-blank answer content. Thinking is separate.
            # Empty content with tool calls gets a short system placeholder.
            has_message = bool(str(content or "").strip())
            existing = self._find_streaming_assistant_entry(turn_meta=turn_meta, create=False)
            if has_message:
                body = str(content)
            elif tool_calls:
                body = f"Requested {len(tool_calls)} tool call(s)."
            else:
                if existing is not None:
                    self._entries.remove(existing)
                return
            is_final = not tool_calls
            if is_final:
                body = _extract_final_answer_body(body)
            if existing is None:
                self._entries.append(
                    ConversationEntry(
                        role="assistant",
                        title="FINAL ANSWER" if is_final else "ASSISTANT",
                        body=body,
                        meta=turn_meta,
                        style="class:final-answer" if is_final else "class:assistant",
                    )
                )
            else:
                existing.title = "FINAL ANSWER" if is_final else "ASSISTANT"
                existing.body = body
                existing.meta = turn_meta
                existing.style = "class:final-answer" if is_final else "class:assistant"
                existing.status = ""
        elif kind == "tool_call_started":
            data = event.data
            name = data.get("name") or "tool"
            tool_id = str(data.get("id") or "")
            self._entries.append(
                ConversationEntry(
                    role="tool-interaction",
                    title=str(name),
                    meta=f"turn {data.get('turn')}" if data.get("turn") else "",
                    style="class:tool",
                    status="running",
                    tool_id=tool_id,
                    arguments=dict(data.get("arguments") or {}),
                )
            )
        elif kind == "tool_result":
            data = event.data
            name = data.get("name") or "tool"
            tool_id = str(data.get("id") or "")
            result = data.get("result") or data.get("error") or event.message
            status = str(data.get("status") or "completed")
            entry = self._find_tool_entry(tool_id=tool_id, name=str(name))
            if entry is None:
                entry = ConversationEntry(
                    role="tool-interaction",
                    title=str(name),
                    style="class:tool-result",
                    tool_id=tool_id,
                    arguments=dict(data.get("arguments") or {}),
                    meta=f"turn {data.get('turn')}" if data.get("turn") else "",
                )
                self._entries.append(entry)
            if data.get("turn"):
                entry.meta = f"turn {data.get('turn')}"
            entry.result = str(result)
            entry.body = str(result)
            entry.extensions = dict(data.get("extensions") or {})
            failed = status in {"failed", "error"}
            entry.status = "failed" if failed else "completed"
            entry.style = "class:error" if failed else "class:tool-result"

    def has_unfinished_thinking_or_tool(self) -> bool:
        return any(
            entry.status == "running" and entry.role in {"thinking", "tool-interaction"}
            for entry in self._entries
        )

    def entries(self) -> Sequence[ConversationEntry]:
        return tuple(self._entries)

    def _find_running_thinking(self, *, turn_meta: str) -> ConversationEntry | None:
        for entry in reversed(self._entries):
            if entry.role == "thinking" and entry.status == "running" and (
                not turn_meta or entry.meta == turn_meta
            ):
                return entry
        return None

    def _find_streaming_assistant_entry(
        self, *, turn_meta: str, create: bool = True
    ) -> ConversationEntry | None:
        for entry in reversed(self._entries):
            if entry.role == "assistant" and entry.status == "streaming" and (
                not turn_meta or entry.meta == turn_meta
            ):
                return entry
        if not create:
            return None
        entry = ConversationEntry(
            role="assistant",
            title="ASSISTANT",
            meta=turn_meta,
            style="class:assistant",
            status="streaming",
        )
        self._entries.append(entry)
        return entry

    def _find_tool_entry(self, *, tool_id: str, name: str) -> ConversationEntry | None:
        for entry in reversed(self._entries):
            if entry.role != "tool-interaction":
                continue
            if tool_id and entry.tool_id == tool_id:
                return entry
            if not tool_id and entry.title == name and entry.status == "running":
                return entry
        return None

    def _complete_latest_thinking(self, *, turn_meta: str, reasoning: str) -> None:
        for entry in reversed(self._entries):
            if entry.role != "thinking" or entry.status != "running":
                continue
            entry.status = "completed"
            entry.title = "thinking completed"
            entry.meta = turn_meta or entry.meta
            if reasoning:
                entry.thinking = reasoning
            entry.style = "class:thinking"
            return
        if reasoning:
            self._entries.append(
                ConversationEntry(
                    role="thinking",
                    title="thinking completed",
                    thinking=reasoning,
                    meta=turn_meta,
                    style="class:thinking",
                    status="completed",
                )
            )
