from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ConversationEntry:
    role: str
    title: str
    body: str = ""
    thinking: str = ""
    meta: str = ""
    style: str = ""
    status: str = ""
    tool_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
