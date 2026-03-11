"""
Web Agent. Based on Alibaba Tongyi DeepResearch repo
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Any, TYPE_CHECKING

from webagent.llm.chat_types import ChatMessage, ToolCall, tool, system, user
from webagent.llm.parser import Parser, ParsingError
from webagent.agent.agent import Agent
from webagent.log import get_logger
from webagent.utils.retry import retry_async, RetryPolicy

if TYPE_CHECKING:
    from webagent.llm.client import Client
    from webagent.tools.tool import Tool

logger = get_logger(__name__)

class WebAgent(Agent):
    """
    Tool-using conversational agent with retry and context-budget safeguards.

    The agent follows an iterative loop:
    1. Send conversation history to the LLM.
    2. Parse the assistant response into chat/tool-call messages.
    3. Execute requested tools and append tool outputs.
    4. Optionally inject reminder prompts near configured turn/token limits.
    5. Stop when no more tool calls are needed or turn budget is exhausted.
    """
    def __init__(self, 
                 llm_client: Client, 
                 parser: Parser, 
                 tools: Iterable[Tool], 
                 system_prompt: str | None = None, 
                 max_turn: int = 10,
                 max_turn_prompt: str | None = None,
                 max_tokens: int = 1024,
                 max_tokens_prompt: str | None = None,
                 max_tokens_prompt_margin: int = 128,
                 llm_retry_policy: RetryPolicy | None = None,
                 tool_retry_policy: RetryPolicy | None = None):
        """
        Initialize a WebAgent instance.

        Args:
            llm_client: LLM client used to generate assistant responses.
            parser: Parser that converts between internal and model message formats.
            tools: Iterable of available tools indexed by name.
            system_prompt: Optional system prompt prepended to every run.
            max_turn: Maximum number of tool-response turns before stopping.
            max_turn_prompt: Optional user prompt injected near turn limit.
            max_tokens: Expected model context limit used for token-budget checks.
            max_tokens_prompt: Optional user prompt injected near token limit.
            max_tokens_prompt_margin: Safety margin before `max_tokens` to trigger
                the token-limit reminder prompt.
            llm_retry_policy: Retry policy for LLM parsing/call steps. If `None`,
                retries are disabled.
            tool_retry_policy: Retry policy for tool execution. If `None`, retries
                are disabled.
        """
        self.client = llm_client
        self.parser = parser
        self.tool_dict = {t.name: t for t in tools}
        self.system_prompt = system_prompt or ""
        self.max_turn = max_turn
        self.max_turn_prompt = max_turn_prompt
        self.context_token_size = 0
        self.max_tokens = max_tokens
        self.max_tokens_prompt = max_tokens_prompt
        self.max_tokens_prompt_margin = max_tokens_prompt_margin
        self.llm_retry_policy = llm_retry_policy
        self.tool_retry_policy = tool_retry_policy

    async def init_tools(self) -> None:
        tools = list(self.tool_dict.values())
        if not tools:
            return
        logger.info("Initializing tools count=%s tools=%s", len(tools), [t.name for t in tools])
        await asyncio.gather(*[t.init() for t in tools])

    async def call_tools(self, tool_calls: Iterable[ToolCall]) -> list[str]:
        tool_call_list = list(tool_calls)
        logger.info("Calling tools count=%s tools=%s", len(tool_call_list), [tc.name for tc in tool_call_list])
        tool_call_coros = []
        for tc in tool_call_list:
            if self.tool_retry_policy is None:
                tool_call_coros.append(self.tool_dict[tc.name].run(**dict(tc.arguments)))
            else:
                tool_call_coros.append(
                    retry_async(
                        self.tool_dict[tc.name].run,
                        policy=self.tool_retry_policy,
                        op_name=f"tool.{tc.name}",
                        log=logger,
                        **dict(tc.arguments),
                    )
                )
            
        return await asyncio.gather(*tool_call_coros)
    
    async def stop(self, history: list[ChatMessage]) -> bool:
        if history[-1].role == "assistant": # no more tool responses
            return True
        if sum(map(lambda x: x.role == "tool", history)) >= self.max_turn:
            return True
        return False
    
    async def parse_and_call_llm(self, history: list[ChatMessage]):
        call_result_raw, usage = await self.client.complete_with_usage(self.parser.to_model(history))
        self.context_token_size = usage.total_tokens if usage else -1
        logger.debug("LLM turn completed total_tokens=%s", self.context_token_size)
        return next(iter(self.parser.from_model([call_result_raw])))

    async def run(self, query: str, extra: dict[str, Any] | None = None):
        """
        Run the agent loop for a single user query.

        Args:
            query: User input task/question.
            extra: Optional extension payload for future customization.

        Returns:
            Full chat history generated during the run, including system/user,
            assistant, and tool messages.
        """
        await self.init_tools()
        history: list[ChatMessage] = [
            system(self.system_prompt, tools=list(self.tool_dict.values())),
            user(query),
        ]
        logger.info("Starting reasoning loop agent=WebAgent query=%r", query[:120])
        turn = 0
        while True:
            turn += 1
            logger.debug("Calling LLM agent=WebAgent turn=%s history_messages=%s", turn, len(history))
            if self.llm_retry_policy is None:
                call_res = await self.parse_and_call_llm(history)
            else:
                call_res = await retry_async(
                    self.parse_and_call_llm,
                    history,
                    policy=self.llm_retry_policy,
                    op_name="webagent.parse_and_call_llm",
                    log=logger,
                )
            
            history.append(call_res)
            
            if call_res.tool_calls:
                results = await self.call_tools(call_res.tool_calls)
                if results:
                    history.append(tool(results))
                    
            if sum(map(lambda x: x.role == "tool", history)) == self.max_turn - 1 and self.max_turn_prompt:
                history.append(user(self.max_turn_prompt))
                
            if self.max_tokens_prompt:
                if self.context_token_size == -1:
                    logger.warning("LLM usage metadata missing; skip max_tokens reminder")
                elif self.context_token_size > self.max_tokens - self.max_tokens_prompt_margin:
                    logger.info(
                        "Context token limit approaching total_tokens=%s limit=%s margin=%s",
                        self.context_token_size,
                        self.max_tokens,
                        self.max_tokens_prompt_margin,
                    )
                    history.append(user(self.max_tokens_prompt))
            
            if await self.stop(history):
                break
            
        logger.info("Reasoning completed agent=WebAgent turns=%s messages=%s", turn, len(history))
        return history
