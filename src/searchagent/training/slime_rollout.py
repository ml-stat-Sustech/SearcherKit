from __future__ import annotations

import copy
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import ValidationError
from omegaconf import OmegaConf
from slime.agent.trajectory import fan_out_sample_segments, merge_turns
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from searchagent.agent.search_agent import LLMContextError, LLMOutputError
from searchagent.llm.parsers import ParsingError
from searchagent.llm.parsers import get_parser
from searchagent.log import get_logger
from searchagent.training.agent import (
    RepeatedToolCallError,
    SearchAgentTraining,
    TooManyToolCallsError,
)
from searchagent.training.config import AgentConfig
from searchagent.training.rewards import assign_overlong_penalty, f1_score
from searchagent.training.slime_client import SlimeSGLangClient

logger = get_logger(__name__)

_ANSWER_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^}]*)\}", re.DOTALL)


def _arg_value(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _select_key(config: Any, key: str) -> Any:
    node = config
    for part in key.split("."):
        node = node[part]
    return node


@lru_cache(maxsize=16)
def _load_agent_config(path_raw: str, key: str) -> AgentConfig:
    path = Path(path_raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SearchAgent config file not found: {path}")
    raw = OmegaConf.load(path)
    selected = _select_key(raw, key)
    merged = OmegaConf.merge(OmegaConf.structured(AgentConfig), selected)
    config = OmegaConf.to_object(merged)
    if not isinstance(config, AgentConfig):
        raise TypeError(f"Expected AgentConfig from {path}:{key}, got {type(config).__name__}")
    return config


def _agent_config_for(args: Any, *, evaluation: bool) -> AgentConfig:
    config_path = _arg_value(args, "searchagent_agent_config")
    if not config_path:
        raise ValueError("--searchagent-agent-config is required for SearchAgent slime rollout")
    key = (
        _arg_value(args, "searchagent_eval_agent_config_key", None)
        if evaluation
        else _arg_value(args, "searchagent_agent_config_key", None)
    )
    key = key or _arg_value(args, "searchagent_agent_config_key", "agent")
    return copy.deepcopy(_load_agent_config(config_path, key))


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _question_from_sample(sample: Sample) -> str:
    prompt = sample.prompt
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return _flatten_content(message.get("content"))
        return "\n".join(
            _flatten_content(message.get("content"))
            for message in prompt
            if isinstance(message, dict)
        )
    return str(prompt)


def _label_from_sample(sample: Sample) -> str:
    if sample.label is not None:
        return str(sample.label)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("answer", "label", "target"):
        if metadata.get(key) is not None:
            return str(metadata[key])
    return ""


def _final_content(history: list[Any]) -> str:
    last_msg = history[-1] if history else None
    if last_msg is None or getattr(last_msg, "role", None) != "assistant":
        return ""
    content = getattr(last_msg, "content", None)
    return content if isinstance(content, str) else ""


def _score_history(
    *,
    history: list[Any],
    label: str,
    context_token_size: int,
    max_tokens: int,
    max_tokens_prompt_margin: int,
    format_error: bool,
) -> tuple[float, dict[str, Any]]:
    last_content = _final_content(history)
    matches = list(_ANSWER_PATTERN.finditer(last_content))
    answer = None
    if len(matches) != 1:
        format_error = True
    else:
        answer = matches[0].group("answer").strip()

    outcome_score = f1_score(answer.lower(), label.lower()) if answer else 0.0
    overlong_penalty = assign_overlong_penalty(
        context_token_size,
        max_tokens,
        max_tokens_prompt_margin / 2,
    )
    reward = outcome_score + overlong_penalty
    return reward, {
        "answer": answer,
        "format_error": format_error,
        "format_score": 0.0 if format_error else 0.1,
        "outcome_score": outcome_score,
        "overlong_penalty": overlong_penalty,
        "context_tokens": context_token_size,
    }


def _empty_sample(sample: Sample, tokenizer: Any, reward: float, metadata: dict[str, Any]) -> Sample:
    prompt = sample.prompt if isinstance(sample.prompt, str) else _question_from_sample(sample)
    sample.tokens = tokenizer.encode(prompt, add_special_tokens=False)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.reward = float(reward)
    sample.status = Sample.Status.FAILED
    sample.metadata = {**(sample.metadata or {}), **metadata}
    return sample


async def generate_searchagent(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """slime ``--custom-generate-function-path`` entry for SearchAgent."""

    state = GenerateState(args)
    agent_config = _agent_config_for(args, evaluation=evaluation)
    client = SlimeSGLangClient(
        args=args,
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        default_kwargs=agent_config.llm_client.default_kwargs,
        use_provider_tools=get_parser(agent_config.parser).uses_provider_tools,
        tool_call_parser=_arg_value(args, "searchagent_tool_call_parser", "qwen"),
        reasoning_parser=_arg_value(args, "searchagent_reasoning_parser", "qwen3"),
        max_context_tokens=agent_config.max_tokens,
        model_name=_arg_value(args, "searchagent_rollout_model_name", "default"),
        session_id=sample.session_id,
    )
    agent = SearchAgentTraining(config=agent_config, llm_client=client)

    format_error = False
    context_error = False
    repeated_query = False
    too_many_tool_call = False
    history: list[Any] = []
    try:
        history = await agent.run(_question_from_sample(sample), session_id=sample.index)
    except RepeatedToolCallError as exc:
        format_error = True
        repeated_query = True
        history = agent.history
        logger.warning(repr(exc))
    except TooManyToolCallsError as exc:
        format_error = True
        too_many_tool_call = True
        history = agent.history
        logger.warning(repr(exc))
    except (LLMOutputError, ValidationError, ParsingError) as exc:
        format_error = True
        history = agent.history
        logger.warning(repr(exc))
    except LLMContextError as exc:
        context_error = True
        history = agent.history
        logger.warning(repr(exc))

    label = _label_from_sample(sample)
    reward, reward_metadata = _score_history(
        history=history,
        label=label,
        context_token_size=agent.context_token_size,
        max_tokens=agent.max_tokens,
        max_tokens_prompt_margin=agent.max_tokens_prompt_margin,
        format_error=format_error,
    )
    metadata = {
        **reward_metadata,
        "label": label,
        "num_turns": agent.turn,
        "context_error": context_error,
        "repeated_query": repeated_query,
        "too_many_tool_call": too_many_tool_call,
    }
    segment = merge_turns(client.turns, metadata=metadata)
    if segment is None or not segment.response_ids:
        return _empty_sample(sample, state.tokenizer, reward, metadata)

    status = Sample.Status.COMPLETED
    if context_error or client.context_truncated:
        status = Sample.Status.TRUNCATED
    elif format_error:
        status = Sample.Status.FAILED

    out = fan_out_sample_segments(
        sample,
        [segment],
        reward,
        state.tokenizer,
        metadata=metadata,
    )
    for item in out:
        item.status = status
    return out[0] if len(out) == 1 else out


async def custom_rm(args: Any, sample: Sample | list[Sample], **kwargs: Any) -> float | list[float]:
    """Fallback slime ``--custom-rm-path`` for already-generated SearchAgent samples."""

    if isinstance(sample, list):
        return [float(await custom_rm(args, item, **kwargs)) for item in sample]

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "outcome_score" in metadata and "overlong_penalty" in metadata:
        return float(metadata["outcome_score"]) + float(metadata["overlong_penalty"])
    label = _label_from_sample(sample)
    matches = list(_ANSWER_PATTERN.finditer(sample.response or ""))
    answer = matches[0].group("answer").strip() if len(matches) == 1 else ""
    return f1_score(answer.lower(), label.lower()) if answer else 0.0


def _sample_reward_value(args: Any, sample: Sample | list[Sample]) -> float:
    if isinstance(sample, list):
        values = [float(item.get_reward_value(args)) for item in sample]
        return sum(values)
    return float(sample.get_reward_value(args))


def mixed_reward_filter(args: Any, samples: list[Sample | list[Sample]], **kwargs: Any) -> DynamicFilterOutput:
    rewards = [_sample_reward_value(args, sample) for sample in samples]
    keep = bool(rewards) and min(rewards) < max(rewards)
    reason = None if keep else f"constant_reward_{round(rewards[0], 3) if rewards else 'empty'}"
    return DynamicFilterOutput(keep=keep, reason=reason)
