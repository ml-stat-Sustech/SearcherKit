from __future__ import annotations

from collections.abc import Sequence

from searcherkit.interfaces.tui.chat.conversation_entry import ConversationEntry
from searcherkit.interfaces.tui.ui.formatting import (
    _body_style,
    _format_document_tree,
    _tool_detail_documents_hint,
    _display_width,
    _format_json_compact,
    _indent_multiline,
    _pad_to_display_width,
    _role_prefix,
    _formatted_lines,
)
from searcherkit.runtime.trace import preview_query

STREAMING_CURSOR = "▌"


class ChatRenderer:
    """Turns a snapshot of chat entries into styled prompt-toolkit display parts."""

    def render_full(
        self,
        entries: Sequence[ConversationEntry],
        *,
        chat_width: int,
        show_thinking: bool,
        show_tool_detail: bool,
    ) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        hidden_reasoning = 0
        for entry in entries:
            reasoning = entry.thinking if show_thinking else ""
            if entry.thinking and not show_thinking:
                hidden_reasoning += 1
            if (
                not entry.body.strip()
                and not reasoning.strip()
                and entry.role in {"assistant", "thinking"}
                and not (entry.title == "FINAL ANSWER" and entry.status == "streaming")
            ):
                continue
            self._append_entry_parts(
                parts,
                entry,
                reasoning=reasoning,
                show_tool_detail=show_tool_detail,
                chat_width=chat_width,
            )
        if hidden_reasoning:
            parts.append(
                ("class:muted", f"[{hidden_reasoning} reasoning block(s) hidden; use Ctrl+O]\n")
            )
        return parts or [("class:muted", " ")]

    def render_plain_lines(self, parts: list[tuple[str, str]]) -> list[str]:
        return ["".join(text for _, text in line) for line in _formatted_lines(parts)]

    def _append_entry_parts(
        self,
        parts: list[tuple[str, str]],
        entry: ConversationEntry,
        *,
        reasoning: str,
        show_tool_detail: bool,
        chat_width: int,
    ) -> None:
        if entry.role == "tool-interaction":
            self._append_tool_entry_parts(parts, entry, show_tool_detail=show_tool_detail, chat_width=chat_width)
            return
        if entry.role == "user":
            self._append_shaded_entry_parts(
                parts,
                entry,
                reasoning=reasoning,
                shadow_style="class:user-shadow",
                title_style=entry.style or "class:user",
                body_style="class:user-body",
                chat_width=chat_width,
            )
            return
        if entry.style == "class:final-answer" or entry.title == "FINAL ANSWER":
            self._append_shaded_entry_parts(
                parts,
                entry,
                reasoning=reasoning,
                shadow_style="class:final-answer-shadow",
                title_style=entry.style or "class:final-answer",
                body_style="class:final-answer-body",
                chat_width=chat_width,
            )
            return
        if entry.role == "assistant":
            # Intermediate assistant message: body only, no rule divider.
            self._append_assistant_body_parts(parts, entry, chat_width=chat_width)
            return
        title = self._entry_title(entry)
        if entry.role != "thinking":
            parts.append((entry.style or "class:meta", f"{self._rule_title(title, chat_width=chat_width)}\n"))
        if reasoning:
            thinking_indent = "  "
            wrapped_thinking = _indent_multiline(
                reasoning,
                width=self._body_wrap_width(chat_width=chat_width, indent=thinking_indent),
                indent=thinking_indent,
            )
            parts.append(("class:reasoning-body", wrapped_thinking))
            if entry.role == "thinking" and entry.status == "running":
                parts.append(("class:streaming-cursor", STREAMING_CURSOR))
            if not wrapped_thinking.endswith("\n"):
                parts.append(("class:reasoning-body", "\n"))
        if entry.body:
            body_style = _body_style(entry.role)
            body_indent = "  "
            wrapped = _indent_multiline(
                entry.body,
                width=self._body_wrap_width(chat_width=chat_width, indent=body_indent),
                indent=body_indent,
            )
            parts.append((body_style, wrapped))
            if not wrapped.endswith("\n"):
                parts.append((body_style, "\n"))
        parts.append(("", "\n"))

    def _append_assistant_body_parts(
        self,
        parts: list[tuple[str, str]],
        entry: ConversationEntry,
        *,
        chat_width: int,
    ) -> None:
        """Render intermediate assistant content without a rule divider."""
        if not entry.body:
            return
        body_indent = "  "
        wrapped = _indent_multiline(
            entry.body,
            width=self._body_wrap_width(chat_width=chat_width, indent=body_indent),
            indent=body_indent,
        )
        parts.append(("class:assistant-body", wrapped))
        if entry.status == "streaming":
            parts.append(("class:streaming-cursor", STREAMING_CURSOR))
        if not wrapped.endswith("\n"):
            parts.append(("class:assistant-body", "\n"))
        parts.append(("", "\n"))

    def _append_shaded_entry_parts(
        self,
        parts: list[tuple[str, str]],
        entry: ConversationEntry,
        *,
        reasoning: str,
        shadow_style: str,
        title_style: str,
        body_style: str,
        chat_width: int,
    ) -> None:
        parts.append((shadow_style, f"{self._block_shadow_line(chat_width=chat_width)}\n"))
        self._append_block_text(
            parts,
            title_style,
            self._rule_title(self._entry_title(entry), chat_width=chat_width),
            chat_width=chat_width,
        )
        if reasoning:
            thinking_indent = "  "
            wrapped_thinking = _indent_multiline(
                reasoning,
                width=self._body_wrap_width(chat_width=chat_width, indent=thinking_indent),
                indent=thinking_indent,
            )
            self._append_block_text(parts, "class:reasoning-body", wrapped_thinking, chat_width=chat_width)
        if entry.body:
            body_indent = "  "
            wrapped = _indent_multiline(
                entry.body,
                width=self._body_wrap_width(chat_width=chat_width, indent=body_indent),
                indent=body_indent,
            )
            self._append_block_text(parts, body_style, wrapped, chat_width=chat_width)
        parts.append((shadow_style, f"{self._block_shadow_line(chat_width=chat_width)}\n"))
        parts.append(("", "\n"))

    def _append_tool_entry_parts(
        self,
        parts: list[tuple[str, str]],
        entry: ConversationEntry,
        *,
        show_tool_detail: bool,
        chat_width: int,
    ) -> None:
        status_label = entry.status or "completed"
        if status_label == "failed":
            title_style = "class:tool-error"
        elif status_label == "running":
            title_style = "class:tool"
        else:
            title_style = "class:tool-result"
        parts.append(("class:tool-shadow", f"{self._block_shadow_line(chat_width=chat_width)}\n"))
        title = "TOOL"
        if entry.meta:
            title = f"{title} · {entry.meta}"
        self._append_block_text(
            parts, title_style, self._rule_title(title, chat_width=chat_width), chat_width=chat_width
        )

        call_indent = "  "
        call_line = self._tool_call_display_line(
            entry,
            max_chars=2000 if show_tool_detail else 220,
        )
        wrapped_call = _indent_multiline(
            call_line,
            width=self._body_wrap_width(chat_width=chat_width, indent=call_indent),
            indent=call_indent,
        )
        self._append_tool_call_block_text(parts, wrapped_call, tool_name=entry.title or "tool", chat_width=chat_width)

        if entry.result:
            document_tree = None if show_tool_detail else _format_document_tree(entry.extensions)
            if document_tree is not None:
                tree_text, document_count = document_tree
                tree_indent = "  "
                wrapped_tree = _indent_multiline(
                    tree_text,
                    width=self._body_wrap_width(chat_width=chat_width, indent=tree_indent),
                    indent=tree_indent,
                )
                body_style = (
                    "class:tool-error-body" if entry.status == "failed" else "class:tool-result-body"
                )
                self._append_block_text(parts, body_style, wrapped_tree, chat_width=chat_width)
                self._append_block_text(
                    parts,
                    "class:tool-muted",
                    f"  {_tool_detail_documents_hint(document_count)}",
                    chat_width=chat_width,
                )
            else:
                parts.append(("class:tool-shadow", f"{self._block_shadow_line(chat_width=chat_width)}\n"))
                result_text = entry.result
                body_style = (
                    "class:tool-error-body" if entry.status == "failed" else "class:tool-result-body"
                )
                result_indent = "  "
                wrapped = _indent_multiline(
                    result_text,
                    width=self._body_wrap_width(chat_width=chat_width, indent=result_indent),
                    indent=result_indent,
                )
                self._append_block_text(parts, body_style, wrapped, chat_width=chat_width)
        parts.append(("class:tool-shadow", f"{self._block_shadow_line(chat_width=chat_width)}\n"))
        parts.append(("", "\n"))

    def _append_block_text(
        self, parts: list[tuple[str, str]], style: str, text: str, *, chat_width: int
    ) -> None:
        width = self._block_shadow_width(chat_width=chat_width)
        for line in text.splitlines() or [""]:
            parts.append((style, f"{_pad_to_display_width(line, width)}\n"))

    def _append_tool_call_block_text(
        self,
        parts: list[tuple[str, str]],
        text: str,
        *,
        tool_name: str,
        chat_width: int,
    ) -> None:
        width = self._block_shadow_width(chat_width=chat_width)
        name_style = "class:tool-name"
        for index, line in enumerate(text.splitlines() or [""]):
            if index == 0:
                name_start = line.find(tool_name)
                if name_start >= 0:
                    name_end = name_start + len(tool_name)
                    parts.append(("class:tool-body", line[:name_start]))
                    parts.append((name_style, line[name_start:name_end]))
                    suffix = line[name_end:]
                    parts.append(
                        (
                            "class:tool-body",
                            _pad_to_display_width(
                                suffix, width - _display_width(line[:name_end])
                            ),
                        )
                    )
                    parts.append(("class:tool-body", "\n"))
                    continue
            parts.append(("class:tool-body", f"{_pad_to_display_width(line, width)}\n"))

    def _block_shadow_line(self, *, chat_width: int) -> str:
        return " " * self._block_shadow_width(chat_width=chat_width)

    def _block_shadow_width(self, *, chat_width: int) -> int:
        return max(1, chat_width - 2)

    def _tool_call_display_line(self, entry: ConversationEntry, *, max_chars: int) -> str:
        tool_name = entry.title or "tool"
        if not entry.arguments:
            return tool_name
        if len(entry.arguments) == 1:
            argument_text = _format_json_compact(next(iter(entry.arguments.values())), max_chars=max_chars)
        else:
            argument_text = ", ".join(
                _format_json_compact(value, max_chars=max_chars)
                for value in entry.arguments.values()
            )
        return f"{tool_name} {argument_text}"

    def _entry_title(self, entry: ConversationEntry) -> str:
        title = entry.title or _role_prefix(entry.role)
        if entry.meta and entry.role in {"assistant", "thinking"}:
            title = f"{title} · {entry.meta}"
        return title

    def _body_wrap_width(self, *, chat_width: int, indent: str) -> int:
        return max(16, chat_width - len(indent) - 2)

    def _rule_title(self, title: str, *, chat_width: int) -> str:
        width = max(28, chat_width - 2)
        cleaned = preview_query(title, max(16, width - 4))
        line = f"── {cleaned} "
        line_width = _display_width(line)
        if line_width >= width:
            return line.rstrip()
        return line + "─" * (width - line_width)
