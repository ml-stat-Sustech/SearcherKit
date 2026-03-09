"""
Agent implementation example. React Agent that returns when no more tool calls are requested
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Any, TYPE_CHECKING

from webagent.llm.chat_types import tool, system, user
from webagent.agent.agent import Agent

if TYPE_CHECKING:
    from webagent.llm.chat_types import ChatMessage, ToolCall
    from webagent.llm.client import Client
    from webagent.llm.parser import Parser
    from webagent.tools.tool import Tool

class ReactAgent(Agent):
    """
    Agent implementation example. React Agent that returns when no more tool calls are requested
    """
    def __init__(self, llm_client: Client, parser: Parser, tools: Iterable[Tool], system_prompt: str | None = None):
        self.client = llm_client
        self.parser = parser
        self.tool_dict = {t.name: t for t in tools}
        self.system_prompt = system_prompt or ""

    async def call_tools(self, tool_calls: Iterable[ToolCall]) -> list[str]:
        return await asyncio.gather(*[self.tool_dict[tc.name].run(**tc.arguments) for tc in tool_calls])
    
    async def stop(self, history: list[ChatMessage]) -> bool:
        if history[-1].role == "assistant": # no more tool responses
            return True
        return False

    async def run(self, query: str, extra: dict[str, Any] | None = None):
        history: list[ChatMessage] = [system(self.system_prompt, tools=list(self.tool_dict.values())),
                                      user(query)]
        while True:
            call_res_raw = await self.client.complete(self.parser.to_model(history))
            
            call_res = next(iter(self.parser.from_model([call_res_raw])))
            
            history.append(call_res)
            
            if call_res.tool_calls:
                results = await self.call_tools(call_res.tool_calls)
                if results:
                    history.append(tool(results))
            
            if await self.stop(history):
                break
            
        return history