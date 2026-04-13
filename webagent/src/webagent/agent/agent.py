from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from webagent.commons.messages import ChatMessage

class Agent:
    @abc.abstractmethod
    async def run(self, query: str, session_id: int | None = None, extra: dict[str, Any] | None = None) -> list[ChatMessage]:
        pass