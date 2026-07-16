from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from searchagent.runtime.trace import preview_query


def _format_json_compact(value: Any, *, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return preview_query(text, max_chars)


def _document_count_label(count: int) -> str:
    unit = "document" if count == 1 else "documents"
    return f"{count} {unit}"


def _document_display_name(document: Mapping[str, Any]) -> str:
    title = document.get("title")
    if isinstance(title, str) and title.strip():
        return title
    document_id = document.get("id")
    if document_id is not None and str(document_id).strip():
        return str(document_id)
    return "(untitled)"


def _format_document_tree(
    extensions: dict[str, Any] | None,
) -> tuple[str, int] | None:
    """Format collapsed search/visit hits as a tree under the tool call line.

    Returns ``(tree_text, document_count)``, or ``None`` when structured
    documents are unavailable.
    """
    if not isinstance(extensions, dict):
        return None
    raw_documents = extensions.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        return None

    documents = [item for item in raw_documents if isinstance(item, Mapping)]
    if not documents:
        return None

    lines: list[str] = []
    last_index = len(documents) - 1
    for index, document in enumerate(documents):
        branch = "└─ " if index == last_index else "├─ "
        lines.append(f"{branch}{_document_display_name(document)}")
    return "\n".join(lines), len(documents)


def _tool_detail_documents_hint(count: int) -> str:
    return f"use /tool-detail to expand details for {_document_count_label(count)}"


def _role_prefix(role: str) -> str:
    if role == "user":
        return "USER"
    if role == "assistant":
        return "ASSISTANT"
    if role == "thinking":
        return "THINKING"
    if role == "intro":
        return "INFO"
    if role == "tool-interaction":
        return "TOOL"
    if role == "tool-call":
        return "CALL"
    if role == "tool-result":
        return "RESULT"
    if role == "system":
        return "SYSTEM"
    if role == "error":
        return "ERROR"
    if role == "meta":
        return "INFO"
    return role.upper()


def _body_style(role: str) -> str:
    if role == "user":
        return "class:user-body"
    if role == "assistant":
        return "class:assistant-body"
    if role == "intro":
        return "class:body"
    if role == "thinking":
        return "class:thinking-body"
    if role == "tool-interaction":
        return "class:tool-result-body"
    if role == "tool-call":
        return "class:tool-body"
    if role == "tool-result":
        return "class:tool-result-body"
    if role == "system":
        return "class:system-body"
    if role == "error":
        return "class:error"
    return "class:body"


_ANSWER_TAG_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _extract_final_answer_body(text: str) -> str:
    match = None
    for match in _ANSWER_TAG_RE.finditer(text):
        pass
    if match is None:
        return text
    return match.group(1).strip()


def _char_display_width(char: str) -> int:
    try:
        from prompt_toolkit.utils import get_cwidth
    except ImportError:
        if unicodedata.combining(char):
            return 0
        return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return max(0, get_cwidth(char))


def _display_width(text: str) -> int:
    return sum(_char_display_width(char) for char in text)


def _pad_to_display_width(text: str, width: int) -> str:
    padding = width - _display_width(text)
    if padding <= 0:
        return text
    return text + " " * padding


def _split_by_display_width(text: str, width: int) -> list[str]:
    if width <= 0:
        return [text]
    result: list[str] = []
    current = ""
    current_width = 0
    for char in text:
        char_width = _char_display_width(char)
        if current and current_width + char_width > width:
            result.append(current)
            current = char
            current_width = char_width
        else:
            current += char
            current_width += char_width
    if current:
        result.append(current)
    return result or [""]


def _wrap_display_width(value: str, *, width: int) -> list[str]:
    if width <= 0:
        return [value]
    tokens = re.findall(r"\S+|\s+", value)
    lines: list[str] = []
    current = ""
    current_width = 0
    pending_space = ""
    pending_space_width = 0
    for token in tokens:
        if token.isspace():
            if current:
                pending_space = token
                pending_space_width = _display_width(token)
            continue

        token_width = _display_width(token)
        prefix = pending_space if current else ""
        prefix_width = pending_space_width if current else 0
        if current and current_width + prefix_width + token_width <= width:
            current += prefix + token
            current_width += prefix_width + token_width
            pending_space = ""
            pending_space_width = 0
            continue
        if current:
            lines.append(current)
            current = ""
            current_width = 0
        pending_space = ""
        pending_space_width = 0
        if token_width <= width:
            current = token
            current_width = token_width
            continue
        pieces = _split_by_display_width(token, width)
        lines.extend(pieces[:-1])
        current = pieces[-1]
        current_width = _display_width(current)

    if current:
        lines.append(current)
    return lines or [""]


def _indent_multiline(value: str, *, width: int, indent: str) -> str:
    result: list[str] = []
    for raw_line in value.splitlines() or [""]:
        if not raw_line.strip():
            result.append(indent.rstrip())
            continue
        result.extend(f"{indent}{line}" for line in _wrap_display_width(raw_line, width=width))
    return "\n".join(result)


def _formatted_lines(parts: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    lines: list[list[tuple[str, str]]] = [[]]
    for style, text in parts:
        chunks = text.split("\n")
        for index, chunk in enumerate(chunks):
            if index > 0:
                lines.append([])
            if chunk:
                lines[-1].append((style, chunk))
    if lines and not lines[-1]:
        lines.pop()
    return lines or [[("", "")]]


def _tui_style() -> Any:
    from prompt_toolkit.styles import Style

    return Style.from_dict(
        {
            "panel-title": "bold",
            "muted": "ansibrightblack",
            "selected": "reverse",
            "separator": "ansibrightblack",
            "scrollbar-track": "ansibrightblack",
            "scrollbar-thumb": "reverse",
            "status": "reverse",
            "status-label": "reverse bold",
            "status-model": "reverse ansicyan bold",
            "status-source": "reverse ansiyellow bold",
            "status-records": "reverse ansibrightblack",
            "slash-menu": "ansibrightblack",
            "slash-selected": "reverse",
            "running-kicker": "ansibrightblack",
            "meta": "ansibrightblack",
            "turn": "ansibrightblack",
            "user-shadow": "bg:#102a38",
            "user": "ansicyan bold bg:#102a38",
            "user-body": "ansicyan bg:#102a38",
            "assistant": "ansigreen bold",
            "assistant-body": "",
            "final-answer-shadow": "bg:#1b2338",
            "final-answer": "ansibrightgreen bold bg:#1b2338",
            "final-answer-body": "ansiwhite bg:#1b2338",
            "thinking": "ansibrightblack italic",
            "thinking-body": "ansibrightblack italic",
            "reasoning-body": "ansibrightblack italic",
            "tool-shadow": "bg:#303030",
            "tool": "ansiyellow bold bg:#303030",
            "tool-body": "ansiyellow bg:#303030",
            "tool-name": "ansiwhite bold bg:#303030",
            "tool-result": "ansimagenta bold bg:#303030",
            "tool-result-body": "ansimagenta bg:#303030",
            "tool-muted": "ansibrightblack bg:#303030",
            "tool-error": "ansired bold bg:#303030",
            "tool-error-body": "ansired bg:#303030",
            "system": "ansibrightblack bold",
            "system-body": "ansibrightblack",
            "error": "ansired bold",
            "body": "",
        }
    )
