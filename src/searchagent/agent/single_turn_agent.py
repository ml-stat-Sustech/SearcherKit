"""
Single-turn conversational agent.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from searchagent.common.messages import ChatMessage, system, user
from searchagent.agent import BaseAgent
from searchagent.log import get_logger, log_context

if TYPE_CHECKING:
    from searchagent.llm.client import Client
    from searchagent.llm.parser import Parser

logger = get_logger(__name__)


class SingleTurnAgent(BaseAgent):
    """
    Minimal conversational agent that performs exactly one LLM completion.
    """

    def __init__(
        self,
        llm_client: Client,
        parser: Parser,
        system_prompt: str | None = None,
    ):
        self.client = llm_client
        self.parser = parser
        self.system_prompt = system_prompt or ""

    async def run(self, query: str, extra: dict[str, Any] | None = None) -> list[ChatMessage]:
        history: list[ChatMessage] = [
            system(self.system_prompt),
            user(query),
        ]
        with log_context(turn=1):
            logger.info("Starting single-turn agent query=%r", query[:120])

            call_res_raw = await self.client.complete(self.parser.to_model(history))
            call_res = next(iter(self.parser.from_model([call_res_raw])))
            history.append(call_res)

            logger.info("Single-turn agent completed messages=%s", len(history))
        return history
