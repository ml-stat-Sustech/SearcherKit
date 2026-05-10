
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from typing import Any, Awaitable, Callable, Dict, Iterable, TYPE_CHECKING
import re

from transformers import AutoTokenizer
from openai.types.completion_usage import CompletionUsage
from jsonschema import ValidationError

from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal import workflow_context

from searchagent.agent import SearchAgent
from searchagent.agent.search_agent import LLMOutputError, LLMContextError
from searchagent.llm.parsers import QwenParser, ParsingError
from searchagent.common.messages import ToolCall
from searchagent.errors import RecoverableError
from searchagent.tools import MCPTool
from searchagent.common.retry import RetryPolicy
from searchagent.llm import Client
from searchagent.log import get_logger

from .config_type import WorkFlowConfig

def should_accept(x):
    # for DAPO dynamic filtering
    return 0 < x["rewards"].mean() < 1

def assign_overlong_penalty(context_len, max_len, margin):
    # soft overlong penalty from DAPO
    if context_len <= (max_len - margin):
        return 0.0
    if context_len > max_len:
        return -1.0
    overlong = context_len - (max_len - margin)
    return float(-overlong) / margin

def f1_score(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)

logger = get_logger(__name__)

class SearchAgentTraining(SearchAgent):
    async def call_tools(self, tool_calls: Iterable[ToolCall]) -> list[str]:
        tool_calls = list(tool_calls)
        if self.training and len(tool_calls) > 1:
            raise LLMOutputError("Too many parallel tool calls")
        return await super().call_tools(tool_calls)

class ARealClient(Client):
    """Wrapper around the AReal LLM Client."""
    def __init__(
        self,
        client: ArealOpenAI,
        *,
        default_kwargs: Dict[str, object] | None = None,
        **extra_client_kwargs
    ) -> None:
        """Initialize an OpenAI chat-completions client wrapper.

        Args:
            model: Model name passed to `chat.completions.create`.
            api_key: OpenAI-compatible API key. Uses environment defaults when `None`.
            base_url: Optional base URL for OpenAI-compatible providers.
            default_kwargs: Default request parameters merged into every `complete` call.
            retry_policy: Retry policy for API requests. If `None`, requests are sent
                without retries.
            concurrency_limit: Maximum number of concurrent LLM requests. `None`
                means no explicit semaphore limit.
        """
        self.client = client
        self.default_kwargs = default_kwargs or {}
        
        self._create_completion: Callable[
            [list[dict[str, Any]], dict[str, Any]],
            Awaitable[Any],
        ] = self._create_completion_no_retry

    async def complete(self, messages: Iterable[dict[str,Any]], **kwargs) -> dict[str,Any]:
        """Send chat messages to the model and return the assistant message object.

        Args:
            messages: OpenAI-format chat messages.
            **kwargs: Per-call request parameters that override `default_kwargs`.

        Returns:
            The first choice message from the API response.
        """
        return (await self.complete_with_usage(messages, **kwargs))[0]

    async def _create_completion_no_retry(
        self,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> Any:
        logger.debug(
            "Submitting LLM completion model=%s messages=%s",
            "AReaL Loaded Model",
            len(messages),
        )
        return await self.client.chat.completions.create(
            model="default",
            messages=messages, # type: ignore
            **payload,
        )

    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        **kwargs,
    ) -> tuple[dict[str, Any], CompletionUsage | None]:
        message_list = messages if isinstance(messages, list) else list(messages)
        payload = {**self.default_kwargs, **kwargs}
        resp = await self._create_completion(message_list, payload)
        logger.debug(
            "LLM completion finished model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            "AReaL Loaded Model",
            getattr(resp.usage, "prompt_tokens", None),
            getattr(resp.usage, "completion_tokens", None),
            getattr(resp.usage, "total_tokens", None),
        )
        return resp.choices[0].message.model_dump(), resp.usage
    

class ARealSearchAgentWorkflow(RolloutWorkflow):
    def __init__(self, config: WorkFlowConfig, reward_discount=1.0, export_style="concat") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.agent.model)
        self.training = config.training
        self.agent_config = config.agent
        self.reward = config.reward
        self.overlong_penalty_margin = config.overlong_penalty_margin
        self.reward_discount = reward_discount
        self.export_style = export_style
        # TODO
        assert config.reward == "f1", "LLM as Judge temporarily unavaliable"

    async def arun_episode(self, engine: InferenceEngine, data: Dict[str, Any]) -> Dict[str, Any] | None | Dict[str, InteractionWithTokenLogpReward]:
        # rollout enging wrapper
        areal_client = ArealOpenAI(
            engine=engine, 
            tokenizer=self.tokenizer,
            engine_max_tokens=self.agent_config.max_tokens,
            tool_call_parser="qwen",
            reasoning_parser="qwen3",
            chat_template_type="concat")
        
        # ==================== Build Agent =======================
        agent_config = self.agent_config
        if not agent_config.thinking:
            logger.warning("AReal dose not support hard thinking switch through openai request. Please add soft switch prompt (i.e. /no_think) to system prompt")
        client = ARealClient(
            client=areal_client,
            default_kwargs={
                "temperature": agent_config.generation.temperature,  # slightly larger
                "max_completion_tokens": agent_config.generation.max_new_tokens,
                # "max_tokens": 131072,
                "top_p": agent_config.generation.top_p,
                "extra_body": {
                    # "top_k": agent_config.generation.top_k,
                    "chat_template_kwargs": {
                        "enable_thinking": agent_config.thinking
                    }
                },
                # "stop": ["<tool_response>","\n<tool_response>"] # TODO: somehow this would crash AReaL OpenAIProxyEngine
            })

        if self.training:
            tool_retry_policy = RetryPolicy(
                exceptions = (RecoverableError, )
            )
        else:
            tool_retry_policy = RetryPolicy(
                exceptions= (RecoverableError, ValidationError)
            )

        if self.training:
            tools = [
                MCPTool(
                    "search",
                    "http://10.32.64.11:8102/mcp/",
                    max_concurrency=512,
                    raise_argument_validation_error=True,
                    transport="streamable-http",
                    raise_on_fatal=True
                ),
                # MCPTool(
                #     "visit",
                #     "http://10.32.64.11:8102/mcp/",
                #     max_concurrency=512,
                #     raise_argument_validation_error=True,
                #     transport="streamable-http",
                #     raise_on_fatal=True
                # ),
            ]
        else:
            tools = [
                MCPTool(
                    "search",
                    "http://10.32.64.11:8300/mcp/",
                    max_concurrency=512,
                    raise_argument_validation_error=False,
                    transport="streamable-http",
                    raise_on_fatal=True
                )
            ]

        agent = SearchAgentTraining(
            llm_client=client,
            parser = QwenParser(
                upstream_parsed=True,
            ),
            tools=tools,
            system_prompt=agent_config.system_prompt,
            max_turn=agent_config.max_turn,
            # max_turn_prompt = None,
            max_turn_prompt=agent_config.max_turn_prompt,
            # max_tokens=131072,
            max_tokens=agent_config.max_tokens,
            max_tokens_prompt=agent_config.max_tokens_prompt,
            # max_tokens_prompt = None,
            max_tokens_prompt_margin=agent_config.max_tokens_prompt_margin,
            llm_retry_policy=None,
            tool_retry_policy=tool_retry_policy,
            training=self.training,
            raise_repeat_tool_call=agent_config.raise_repeat_tool_call
        )

        # ===================== Running & Error Handling =======================
        final_reward = None

        format_error = False
        try:
            await agent.run(agent_config.question_prompt.format(Question = data["question"]))
            
            stats_tracker.get(workflow_context.stat_scope()).scalar(context_error=0.0)
            stats_tracker.get(workflow_context.stat_scope()).scalar(repeated_query=0.0)
            stats_tracker.get(workflow_context.stat_scope()).scalar(too_many_tool_call=0.0)
        except (LLMOutputError, ValidationError, ParsingError) as e:
            # `LLMOutputError` from errornous tool name or repeated tool call (if raise_repeat_tool_call=True)
            # `ValidationError` from errornous tool argument schema
            # `ParsingError` from errornous tool call json string
            format_error = True
            stats_tracker.get(workflow_context.stat_scope()).scalar(context_error=0.0)
            if "repeated" in repr(e):
                logger.warning(repr(e))
                stats_tracker.get(workflow_context.stat_scope()).scalar(repeated_query=1.0)
            if "tool calls" in repr(e):
                logger.warning(repr(e))
                stats_tracker.get(workflow_context.stat_scope()).scalar(too_many_tool_call=1.0)
            else:
                stats_tracker.get(workflow_context.stat_scope()).scalar(repeated_query=0.0)
        except LLMContextError:
            # raise when the last request to client exceeds openai server side context limit
            stats_tracker.get(workflow_context.stat_scope()).scalar(repeated_query=0.0)
            stats_tracker.get(workflow_context.stat_scope()).scalar(too_many_tool_call=0.0)
            stats_tracker.get(workflow_context.stat_scope()).scalar(context_error=1.0)
            raise
            # final_reward = -1.0 # max context

        history = agent.history
        
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_turns=agent.turn, 
            context_tokens=agent.context_token_size,
            context_limit_prompted=agent.max_token_reminder_prompted)
        
        
        # ======================  Reward Assignment  =======================
        # extract valid answer and determine format error
        visit_cnt = 0
        for msg in history:
            if msg.role == "assistant" and msg.thinking and "<tool_call>" in msg.thinking: # for qwen3-thinking, this happens
                format_error = True
            if msg.role == "assistant":
                for tool in (msg.tool_calls or []):
                    if tool.name == "visit":
                        visit_cnt += 1
                    
        stats_tracker.get(workflow_context.stat_scope()).scalar(visit_cnt = visit_cnt)

        last_msg = history[-1]
        if last_msg.role != "assistant" or not last_msg.content:
            format_error = True
            last_content = ""
            # return 0.0
        else:
            last_content = last_msg.content
        _ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)

        matches = list(_ANSWER_PATTERN.finditer(last_content))
        answer = None
        if len(matches) != 1:
            format_error = True
        else:
            answer = matches[0].group("answer").strip()

        # assign rewards
        format_score = 0.0 if format_error else 0.1 # 0.1 format reward (answer & tool calls)
        overlong_penalty = assign_overlong_penalty(agent.context_token_size, agent.max_tokens, agent.max_tokens_prompt_margin / 2) # set overlong penalty margin half the prompting margin so we don't punish every traj that's been prompted
        outcome_score = f1_score(answer.lower(), data["answer"].lower()) if answer else 0.0

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            format_score=format_score, 
            overlong_penalty=overlong_penalty,
            outcome_score=outcome_score)

        # final_reward = format_score + overlong_penalty + outcome_score
        final_reward = outcome_score + overlong_penalty
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=final_reward)
        
        # ===================== Submit Trajectory =======================
        areal_client.set_last_reward(final_reward)
        areal_client.apply_reward_discount(self.reward_discount)

        return areal_client.export_interactions(style=self.export_style)
