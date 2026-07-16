from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from searcherkit.common.messages import ChatMessage

class BaseAgent(abc.ABC):
    @abc.abstractmethod
    async def run(self, query: str, session_id: int | None = None, extra: dict[str, Any] | None = None) -> list[ChatMessage]:
        pass