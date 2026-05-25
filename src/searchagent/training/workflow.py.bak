from __future__ import annotations

import re
from typing import Any

from jsonschema import ValidationError
from transformers import AutoTokenizer

from areal import workflow_context
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import stats_tracker

from searchagent.agent.search_agent import LLMContextError, LLMOutputError
from searchagent.common.retry import RetryPolicy
from searchagent.errors import RecoverableError
from searchagent.llm.parsers import ParsingError, QwenParser
from searchagent.log import get_logger
from searchagent.sources.factory import build_source
from searchagent.tools import BaseTool, build_tool
from searchagent.tools.search import SearchTool
from searchagent.training.agent import (
    RepeatedToolCallError,
    SearchAgentTraining,
    TooManyToolCallsError,
)
from searchagent.training.areal_client import ARealClient
from searchagent.training.config import AgentConfig, WorkFlowConfig
from searchagent.training.rewards import assign_overlong_penalty, f1_score

logger = get_logger(__name__)

_ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)


def _build_tools(agent_config: AgentConfig) -> list[BaseTool]:
    if agent_config.tools:
        return [build_tool(tool_config) for tool_config in agent_config.tools]
    source = build_source(config=agent_config.source)
    return [SearchTool(source=source)]


class ARealSearchAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        config: WorkFlowConfig,
        reward_discount: float = 1.0,
        export_style: str = "concat",
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.agent.model)
        self.training = config.training
        self.agent_config = config.agent
        self.reward = config.reward
        self.overlong_penalty_margin = config.overlong_penalty_margin
        self.reward_discount = reward_discount
        self.export_style = export_style
        if config.reward != "f1":
            raise ValueError("LLM as Judge is temporarily unavailable")

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        areal_client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            engine_max_tokens=self.agent_config.max_tokens,
            tool_call_parser="qwen",
            reasoning_parser="qwen3",
            chat_template_type="concat",
        )

        agent_config = self.agent_config
        if not agent_config.thinking:
            logger.warning(
                "AReaL does not support hard thinking switch through openai "
                "request. Please add soft switch prompt (i.e. /no_think) to "
                "system prompt"
            )
        client = ARealClient(
            client=areal_client,
            default_kwargs={
                "temperature": agent_config.generation.temperature,
                "max_completion_tokens": agent_config.generation.max_new_tokens,
                "top_p": agent_config.generation.top_p,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": agent_config.thinking,
                    }
                },
            },
        )

        if self.training:
            tool_retry_policy = RetryPolicy(exceptions=(RecoverableError,))
        else:
            tool_retry_policy = RetryPolicy(exceptions=(RecoverableError, ValidationError))

        tools = _build_tools(agent_config)

        agent = SearchAgentTraining(
            llm_client=client,
            parser=QwenParser(upstream_parsed=True),
            tools=tools,
            system_prompt=agent_config.system_prompt,
            query_prompt=agent_config.query_prompt,
            max_turn=agent_config.max_turn,
            max_turn_prompt=agent_config.max_turn_prompt,
            max_tokens=agent_config.max_tokens,
            max_tokens_prompt=agent_config.max_tokens_prompt,
            max_tokens_prompt_margin=agent_config.max_tokens_prompt_margin,
            llm_retry_policy=None,
            tool_retry_policy=tool_retry_policy,
            raise_repeat_tool_call=agent_config.raise_repeat_tool_call,
        )

        format_error = False
        context_error = False
        repeated_query = False
        too_many_tool_call = False
        try:
            agent.reset()
            await agent.run(data["question"])
        except RepeatedToolCallError as exc:
            format_error = True
            repeated_query = True
            logger.warning(repr(exc))
        except TooManyToolCallsError as exc:
            format_error = True
            too_many_tool_call = True
            logger.warning(repr(exc))
        except (LLMOutputError, ValidationError, ParsingError) as exc:
            format_error = True
            logger.warning(repr(exc))
        except LLMContextError:
            context_error = True
            raise
        finally:
            stats = stats_tracker.get(workflow_context.stat_scope())
            stats.scalar(context_error=float(context_error))
            stats.scalar(repeated_query=float(repeated_query))
            stats.scalar(too_many_tool_call=float(too_many_tool_call))

        history = agent.history

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_turns=agent.turn,
            context_tokens=agent.context_token_size,
            context_limit_prompted=agent.max_token_reminder_prompted,
        )

        visit_cnt = 0
        for msg in history:
            if msg.role == "assistant" and msg.thinking and "<tool_call>" in msg.thinking:
                format_error = True
            if msg.role == "assistant":
                for tool in msg.tool_calls or []:
                    if tool.name == "visit":
                        visit_cnt += 1

        stats_tracker.get(workflow_context.stat_scope()).scalar(visit_cnt=visit_cnt)

        last_msg = history[-1] if history else None
        if last_msg is None or last_msg.role != "assistant" or not last_msg.content:
            format_error = True
            last_content = ""
        else:
            last_content = last_msg.content

        matches = list(_ANSWER_PATTERN.finditer(last_content))
        answer = None
        if len(matches) != 1:
            format_error = True
        else:
            answer = matches[0].group("answer").strip()

        format_score = 0.0 if format_error else 0.1
        overlong_penalty = assign_overlong_penalty(
            agent.context_token_size,
            agent.max_tokens,
            agent.max_tokens_prompt_margin / 2,
        )
        outcome_score = f1_score(answer.lower(), data["answer"].lower()) if answer else 0.0

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            format_score=format_score,
            overlong_penalty=overlong_penalty,
            outcome_score=outcome_score,
        )

        final_reward = outcome_score + overlong_penalty
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=final_reward)

        areal_client.set_last_reward(final_reward)
        areal_client.apply_reward_discount(self.reward_discount)

        return areal_client.export_interactions(style=self.export_style)
