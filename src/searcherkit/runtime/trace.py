from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from searcherkit.common.messages import ChatMessage


def preview_query(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def serialize_message(message: Any) -> Any:
    if is_dataclass(message):
        return asdict(message)
    return message


def history_stats(history: list[ChatMessage]) -> dict[str, int]:
    turns = 0
    tool_calls = 0
    tool_messages = 0

    for message in history:
        if message.role == "assistant":
            turns += 1
            message_tool_calls = message.tool_calls or []
            tool_calls += len(message_tool_calls)
        elif message.role == "tool":
            tool_messages += 1

    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "tool_messages": tool_messages,
    }


def tool_summary(history: list[ChatMessage]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for message in history:
        if message.role != "assistant":
            continue
        for tool_call in message.tool_calls or []:
            summary[tool_call.name] = summary.get(tool_call.name, 0) + 1
    return summary


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:6]}"


def make_trace_id() -> str:
    return uuid4().hex[:12]

