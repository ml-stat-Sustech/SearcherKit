"""Search agent implementation based on Alibaba Tongyi DeepResearch patterns."""

from __future__ import annotations

import asyncio
import json
import traceback
import copy
from dataclasses import dataclass, field
from typing import Iterable, Any, TYPE_CHECKING, overload

from openai import BadRequestError, InternalServerError

from searchagent.common.messages import ChatMessage, ToolCall, tool, system, user
from searchagent.common.messages import Tool as ToolMsgType
from searchagent.tools import BaseTool, ToolConfig, build_tool
from searchagent.sources import DataSource, SourceConfig, add_source_cfg
from searchagent.llm.parsers import Parser, ParsingError, ParserConfig, get_parser, QwenParserConfig
from searchagent.llm.base import Client, ClientConfig, get_client, OpenAIConfig
from searchagent.agent import BaseAgent
from searchagent.errors import LLMError
from searchagent.common.log import get_logger, log_context, LogTiming
from searchagent.common.retry import retry_async, RetryPolicy, RetryConfig

# TODO
class LLMOutputError(LLMError):
    """Raised when there's a problem with the LLM output."""
    pass

class LLMContextError(LLMError):
    """Raised when llm exceeds context limit."""
    pass

if TYPE_CHECKING:
    from searchagent.llm.base import Client

logger = get_logger(__name__)


def _preview_payload(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

@dataclass
class SearchAgentConfig:
    llm_client: ClientConfig = field(default_factory=lambda: ClientConfig(
        type="openai",
        model="",
        openai=OpenAIConfig(),
    ))
    parser: ParserConfig = field(default_factory=lambda: ParserConfig(
        type="qwen",
        qwen=QwenParserConfig(),
    ))
    sources: list[SourceConfig] = field(default_factory=list)
    tools: list[ToolConfig] = field(default_factory=list)
    system_prompt: str | None = None
    query_prompt: str | None = None
    max_turn: int = 10
    max_turn_prompt: str | None = None
    max_tokens: int = 1024
    max_tokens_prompt: str | None = None
    max_tokens_prompt_margin: int = 128
    llm_retry_config: RetryConfig | None = None
    tool_retry_config: RetryConfig | None = None

class SearchAgent(BaseAgent):
    """
    Tool-using conversational agent with retry and context-budget safeguards.

    The agent follows an iterative loop:
    1. Send conversation history to the LLM.
    2. Parse the assistant response into chat/tool-call messages.
    3. Execute requested tools and append tool outputs.
    4. Optionally inject reminder prompts near configured turn/token limits.
    5. Stop when no more tool calls are needed or turn budget is exhausted.
    """
    @overload
    def __init__(self, *, config: SearchAgentConfig): ...

    @overload
    def __init__(self, 
                 llm_client: Client, 
                 parser: Parser, 
                 tools: Iterable[BaseTool], 
                 system_prompt: str | None = None, 
                 query_prompt: str | None = None,
                 max_turn: int = 10,
                 max_turn_prompt: str | None = None,
                 max_tokens: int = 1024,
                 max_tokens_prompt: str | None = None,
                 max_tokens_prompt_margin: int = 128,
                 llm_retry_policy: RetryPolicy | None = None,
                 tool_retry_policy: RetryPolicy | None = None):
        """
        Initialize a SearchAgent instance.

        Args:
            llm_client: LLM client used to generate assistant responses.
            parser: Parser that converts between internal and model message formats.
            tools: Iterable of tools used by agent.
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
        ...
    
    def __init__(self, 
                 llm_client: Client | None = None, 
                 parser: Parser | None = None, 
                 tools: Iterable[BaseTool] = [], 
                 system_prompt: str | None = None, 
                 query_prompt: str | None = None,
                 max_turn: int = 10,
                 max_turn_prompt: str | None = None,
                 max_tokens: int = 1024,
                 max_tokens_prompt: str | None = None,
                 max_tokens_prompt_margin: int = 128,
                 llm_retry_policy: RetryPolicy | None = None,
                 tool_retry_policy: RetryPolicy | None = None,
                 *,
                 config: SearchAgentConfig | None = None):
        if config:
            client = get_client(config.llm_client)
            parser = get_parser(config.parser)
            for source_cfg in config.sources:
                add_source_cfg(source_cfg.name, source_cfg)
            tools = [
                build_tool(tool_cfg)
                for tool_cfg in config.tools
            ]
            if config.llm_retry_config:
                llm_retry_policy = RetryPolicy(config=config.llm_retry_config)
            if config.tool_retry_config:
                tool_retry_policy = RetryPolicy(config=config.tool_retry_config)
            self.__init__(
                llm_client=client,
                parser=parser,
                tools=tools,
                system_prompt=config.system_prompt,
                query_prompt=config.query_prompt,
                max_turn=config.max_turn,
                max_turn_prompt=config.max_turn_prompt,
                max_tokens=config.max_tokens,
                max_tokens_prompt=config.max_tokens_prompt,
                max_tokens_prompt_margin=config.max_tokens_prompt_margin,
                llm_retry_policy=llm_retry_policy,
                tool_retry_policy=tool_retry_policy,
            )
            return

        assert llm_client
        assert parser
            
        self.client = llm_client
        self.parser = parser
        self.tool_dict: dict[str, BaseTool] = {}
        for t in tools:
            self._add_tool(t)
        self.system_prompt = system_prompt or ""
        self.query_prompt = query_prompt or "{query}"
        self.max_turn = max_turn
        self.max_turn_prompt = max_turn_prompt
        self.context_token_size = 0
        self.max_tokens = max_tokens
        self.max_tokens_prompt = max_tokens_prompt
        self.max_tokens_prompt_margin = max_tokens_prompt_margin
        self.llm_retry_policy = llm_retry_policy
        self.tool_retry_policy = tool_retry_policy
        self.context_max_token_exceeded = False
        self.history = []

    def _add_tool(self, tool: BaseTool) -> None:
        if tool.name in self.tool_dict:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self.tool_dict[tool.name] = tool
        
    def reset(self):
        self.history = []
        self.context_token_size = 0
        
    @property
    def turn(self):
        return sum(map(lambda x: x.role == "assistant", self.history))

    @property
    def timing_report(self) -> dict[str, Any]:
        return self._timing.to_dict()

    @staticmethod
    def _is_context_length_error(exc: BadRequestError | InternalServerError) -> bool:
        response = getattr(exc, "response", None)
        body = getattr(response, "json", None)
        error_payload = {}
        if callable(body):
            try:
                error_payload = body()
            except (TypeError, ValueError):
                error_payload = {}
        error = error_payload.get("error", {}) if isinstance(error_payload, dict) else {}
        message = str(error.get("message", "")).lower()
        code = str(error.get("code", "")).lower()
        param = str(error.get("param", "")).lower()
        response_text = str(getattr(response, "text", "") or "").lower()
        text = " ".join(
            part for part in [message, code, param, response_text, str(exc).lower()] if part
        )
        return any(
            marker in text
            for marker in (
                "context length",
                "maximum context length",
                "max context length",
                "context window",
                "context_length_exceeded",
                "too many tokens",
            )
        )

    async def init_tools(self) -> None:
        tools = list(self.tool_dict.values())
        if not tools:
            return
        logger.info("Initializing tools count=%s tools=%s", len(tools), [t.name for t in tools])
        await asyncio.gather(*[t.init() for t in tools])

    async def close_tools(self) -> None:
        tools = list(self.tool_dict.values())
        if not tools:
            return
        logger.info("Closing tools count=%s tools=%s", len(tools), [t.name for t in tools])
        results = await asyncio.gather(*[t.close() for t in tools], return_exceptions=True)
        for tool_item, result in zip(tools, results):
            if isinstance(result, Exception):
                logger.error(
                    "Tool close failed name=%s error=%r",
                    tool_item.name,
                    result,
                )

    async def close(self) -> None:
        await self.close_tools()

    async def call_tools(self, tool_calls: Iterable[ToolCall]) -> list[str]:
        
        tool_call_list = list(tool_calls)
        logger.info("Calling tools count=%s tools=%s", len(tool_call_list), [tc.name for tc in tool_call_list])
        tool_call_coros = []

        async def _timed_run(tc_name: str, coro):
            with self._timing(f"tool.{tc_name}"):
                result = await coro
            return result

        for tc in tool_call_list:
            logger.info(
                "Dispatching tool call id=%s name=%s args=%s",
                tc.id,
                tc.name,
                _preview_payload(dict(tc.arguments)),
            )
            async def return_error(name):
                return f"Error: Tool {name} not found"
            if tc.name not in self.tool_dict:
                tool_call_coros.append(return_error(tc.name))
                continue
            
            if self.tool_retry_policy is None:
                tool_call_coros.append(_timed_run(
                    tc.name,
                    self.tool_dict[tc.name].run(**dict(tc.arguments)),
                ))
            else:
                tool_call_coros.append(_timed_run(
                    tc.name,
                    retry_async(
                        self.tool_dict[tc.name].run,
                        policy=self.tool_retry_policy,
                        op_name=f"tool.{tc.name}",
                        log=logger,
                        **dict(tc.arguments),
                    ),
                ))
            
        gathered = await asyncio.gather(*tool_call_coros, return_exceptions=True)
        first_exception: Exception | None = None
        results: list[str] = []
        for tc, result in zip(tool_call_list, gathered):
            if isinstance(result, Exception):
                if first_exception is None:
                    first_exception = result
                logger.error(
                    "Tool response id=%s name=%s failed error=%r",
                    tc.id,
                    tc.name,
                    result,
                )
                continue

            response_preview = _preview_payload(result)
            logger.info(
                "Tool response id=%s name=%s response=%s",
                tc.id,
                tc.name,
                response_preview,
            )
            results.append(result)
        if first_exception is not None:
            raise first_exception
        return results
    
    async def stop(self) -> bool:
        if self.no_more_tool_calls:
            logger.info(
                "Agent loop naturally ends without more tool calls",
            )
            return True
        
        if self.context_max_token_exceeded:
            if self.max_tokens_prompt and not self.max_token_reminder_prompted:
                return False
            logger.info(
                "Stopping agent loop due to model context limit, total tokens=%d", 
                self.context_token_size,
            )
            return True
        
        if self.turn_limit_exceeded:
            if self.max_turn_prompt and not self.max_turn_reminder_prompted:
                return False
            logger.info(
                "Stopping agent loop due to agent turn limit, total turns=%d", 
                self.turn + 1
            )
            return True
        
        return False
    
    async def parse_and_call_llm(self, history: list[ChatMessage]):
        # TODO: better implementation. 
        if self.parser.uses_provider_tools:
            tools = [tool.as_openai_tool() for tool in self.tool_dict.values()]
        else:
            tools = []
        
        parsed = self.parser.to_model(history)

        with self._timing("llm_call"):
            call_result_raw, usage = await self.client.complete_with_usage(
                parsed, tools=tools, session_id=self.id
            )

        self.context_token_size = usage.total_tokens if usage else -1
        logger.debug("LLM turn completed total_tokens=%s", self.context_token_size)

        return next(iter(self.parser.from_model([call_result_raw])))

    async def run(self, query: str, session_id: int | None = None, extra: dict[str, Any] | None = None):
        """
        Run the agent loop for a single user query.

        Args:
            query: User input task/question.
            extra: Optional extension payload for future customization.

        Returns:
            Full chat history generated during the run, including system/user,
            assistant, and tool messages.
        """
        try:
            self._timing = LogTiming()
            with self._timing("total_time"):
                await self.init_tools()
                self.reset()
                self.id = session_id
                self.history: list[ChatMessage] = [
                    system(
                        self.system_prompt,
                        tools=[ToolMsgType(tool.name, tool.description, tool.inputSchema)
                               for tool in self.tool_dict.values()],
                    ),
                    user(self.query_prompt.format(query=query)),
                ]
                logger.info("Starting agent loop query=%r", query[:120])

                self.max_turn_reminder_prompted = False
                self.max_token_reminder_prompted = False
                while True:
                    with self._timing("turn"), log_context(turn=self.turn):

                        logger.debug("Calling LLM turn=%s history_messages=%s", self.turn, len(self.history))

                        # 1. Reset stop flags
                        self.context_max_token_exceeded = False
                        self.turn_limit_exceeded = False
                        self.no_more_tool_calls = False # No more tool call is set to True if llm call is valid (no exceed context length) and contains no tool calls

                        new_call_result = None
                        new_tool_results = None


                        # 2. Execute agent turn and set stop flags
                        try:
                            if self.llm_retry_policy is None:
                                new_call_result = await self.parse_and_call_llm(self.history)
                            else:
                                new_call_result = await retry_async(
                                    self.parse_and_call_llm,
                                    self.history,
                                    policy=self.llm_retry_policy,
                                    op_name="searchagent.parse_and_call_llm",
                                    log=logger,
                                )
                        except (BadRequestError, InternalServerError) as exc:
                            if self._is_context_length_error(exc):
                                logger.warning("LLM context length error, stop agent loop, context token=%s", self.context_token_size)
                                raise LLMContextError from exc
                            traceback.print_exc()
                            raise

                        if self.turn >= self.max_turn - 1 and new_call_result.tool_calls:
                            self.turn_limit_exceeded = True

                        if self.context_token_size >= self.max_tokens - self.max_tokens_prompt_margin:
                            self.context_max_token_exceeded = True

                        # call tools from llm result (if any)
                        if new_call_result.tool_calls:
                            results = await self.call_tools(new_call_result.tool_calls)
                            for tc, r in zip(new_call_result.tool_calls, results):
                                tc.result = r
                            new_tool_results = tool([copy.copy(tc) for tc in new_call_result.tool_calls])
                        else:
                            self.no_more_tool_calls = True


                        # 3. Decide stop
                        if await self.stop():
                            # add this turn msg and stop
                            if new_call_result:
                                self.history.append(new_call_result)
                                if new_tool_results:
                                    self.history.append(new_tool_results)
                            logger.info(
                                "Agent loop now stopped",
                            )
                            break


                        # 4. A chance to wrap up before limit encontered
                        if self.context_max_token_exceeded and not self.max_token_reminder_prompted and self.max_tokens_prompt:
                            logger.warning(
                                "Context limit apporaching, total=%d, limit=%d, margin=%d triggered, Requesting model to wrap up",
                                self.context_token_size,
                                self.max_tokens,
                                self.max_tokens_prompt_margin,
                            )
                            self.history.append(new_call_result)
                            new_tool_results.tool_responses = new_tool_results.tool_responses[:1]
                            new_tool_results.tool_responses[0].result = self.max_tokens_prompt
                            self.history.append(new_tool_results)
                            self.max_token_reminder_prompted = True
                            continue
                        if self.turn_limit_exceeded and not self.max_turn_reminder_prompted and self.max_turn_prompt:
                            logger.warning(
                                "Turn limit apporaching, turn=%d, limit=%d, Requesting model to wrap up",
                                self.turn,
                                self.max_turn,
                            )
                            self.history.append(new_call_result)
                            # self.history.append(user(self.max_turn_prompt))
                            new_tool_results.tool_responses = new_tool_results.tool_responses[:1]
                            new_tool_results.tool_responses[0].result = self.max_turn_prompt
                            self.history.append(new_tool_results)
                            self.max_turn_reminder_prompted = True
                            continue


                        # 5. Append msg of this turn. Turn number would not be increased before here
                        if new_call_result:
                            self.history.append(new_call_result)
                            assert new_tool_results # tool results should be set when reached here
                            self.history.append(new_tool_results)

            logger.info("Reasoning completed agent=SearchAgent turns=%s messages=%s", self.turn, len(self.history))
            return self.history
        finally:
            await self.close_tools()
