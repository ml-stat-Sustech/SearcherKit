"""
Agent implementation example. React Agent that returns when no more tool calls are requested
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Any, TYPE_CHECKING

from webagent.llm.chat_types import tool, system, user
from webagent.agent.agent import Agent
from webagent.log import get_logger

if TYPE_CHECKING:
    from webagent.llm.chat_types import ChatMessage, ToolCall
    from webagent.llm.client import Client
    from webagent.llm.parser import Parser
    from webagent.tools.tool import Tool

logger = get_logger(__name__)

class ReactAgent(Agent):
    """
    Agent implementation example. React Agent that returns when no more tool calls are requested
    """
    def __init__(self, llm_client: Client, parser: Parser, tools: Iterable[Tool], system_prompt: str | None = None):
        self.client = llm_client
        self.parser = parser
        self.tool_dict = {t.name: t for t in tools}
        self.system_prompt = system_prompt or ""

    async def init_tools(self) -> None:
        tools = list(self.tool_dict.values())
        if not tools:
            return
        logger.info("Initializing tools count=%s tools=%s", len(tools), [t.name for t in tools])
        await asyncio.gather(*[t.init() for t in tools])

    async def call_tools(self, tool_calls: Iterable[ToolCall]) -> list[str]:
        tool_call_list = list(tool_calls)
        logger.info("Calling tools count=%s tools=%s", len(tool_call_list), [tc.name for tc in tool_call_list])
        return await asyncio.gather(*[self.tool_dict[tc.name].run(**tc.arguments) for tc in tool_call_list])
    
    async def stop(self, history: list[ChatMessage]) -> bool:
        if history[-1].role == "assistant": # no more tool responses
            return True
        return False

    async def run(self, query: str, extra: dict[str, Any] | None = None):
        await self.init_tools()
        history: list[ChatMessage] = [
            system(self.system_prompt, tools=list(self.tool_dict.values())),
            user(query),
        ]
        logger.info("Starting reasoning loop agent=ReactAgent query=%r", query[:120])
        turn = 0
        while True:
            turn += 1
            logger.debug("Running LLM turn=%s history_messages=%s", turn, len(history))
            call_res_raw = await self.client.complete(self.parser.to_model(history))
            
            call_res = next(iter(self.parser.from_model([call_res_raw])))
            
            history.append(call_res)
            
            if call_res.tool_calls:
                results = await self.call_tools(call_res.tool_calls)
                if results:
                    history.append(tool(results))
            
            if await self.stop(history):
                break
            
        logger.info("Reasoning completed agent=ReactAgent turns=%s messages=%s", turn, len(history))
        return history
