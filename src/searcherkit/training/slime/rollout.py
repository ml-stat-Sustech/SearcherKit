from __future__ import annotations

import asyncio
import copy
import re
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from jsonschema import ValidationError
from omegaconf import OmegaConf
from slime.agent.trajectory import fan_out_sample_segments, merge_turns
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.sglang_rollout import GenerateState, get_model_url
from slime.utils import logging_utils
from slime.utils.http_utils import post
from slime.utils.metric_utils import (
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
)
from slime.utils.types import Sample

from searcherkit.agent.search_agent import LLMContextError, LLMOutputError
from searcherkit.llm.parsers import ParsingError
from searcherkit.llm.parsers import get_parser
from searcherkit.common.log import get_logger
from searcherkit.training.agent import (
    RepeatedToolCallError,
    SearchAgentTraining,
    TooManyToolCallsError,
)
from searcherkit.training.config import AgentConfig
from searcherkit.training.rewards import (
    assign_overlong_penalty,
    count_duplicate_tool_results,
    count_repeated_tool_queries,
    count_tool_calls,
    count_truncated_tool_responses,
    f1_score,
    searcherkit_reward_components,
)
from searcherkit.training.slime.client import SlimeSGLangClient

logger = get_logger(__name__)

_ANSWER_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^}]*)\}", re.DOTALL)
_DEFAULT_EVAL_AGENT_TIMEOUT_SECONDS = 7200.0


def _arg_value(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _answer_pattern(args: Any) -> re.Pattern[str]:
    return re.compile(_arg_value(args, "searcherkit_answer_pattern", _ANSWER_PATTERN.pattern), re.DOTALL)


def _select_key(config: Any, key: str) -> Any:
    node = config
    for part in key.split("."):
        node = node[part]
    return node


@lru_cache(maxsize=16)
def _load_agent_config(path_raw: str, key: str) -> AgentConfig:
    path = Path(path_raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SearcherKit config file not found: {path}")
    raw = OmegaConf.load(path)
    selected = _select_key(raw, key)
    merged = OmegaConf.merge(OmegaConf.structured(AgentConfig), selected)
    config = OmegaConf.to_object(merged)
    if not isinstance(config, AgentConfig):
        raise TypeError(f"Expected AgentConfig from {path}:{key}, got {type(config).__name__}")
    return config


def _agent_config_for(args: Any, *, evaluation: bool) -> AgentConfig:
    config_path = _arg_value(args, "searcherkit_agent_config")
    if not config_path:
        raise ValueError("--searcherkit-agent-config is required for SearcherKit slime rollout")
    key = (
        _arg_value(args, "searcherkit_eval_agent_config_key", None)
        if evaluation
        else _arg_value(args, "searcherkit_agent_config_key", None)
    )
    key = key or _arg_value(args, "searcherkit_agent_config_key", "agent")
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
    truncation_penalty: float,
    format_error: bool,
    repeated_query: bool,
    too_many_tool_call: bool,
    answer_pattern: re.Pattern[str],
    truncated: bool = False,
) -> tuple[float, dict[str, Any]]:
    last_content = _final_content(history)
    matches = list(answer_pattern.finditer(last_content))
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
    tool_call_count = count_tool_calls(history)
    reward_parts = searcherkit_reward_components(
        outcome_score=outcome_score,
        overlong_penalty=overlong_penalty,
        truncation_penalty=truncation_penalty if truncated else 0.0,
        format_error=format_error,
        tool_call_count=tool_call_count,
        repeated_query=repeated_query,
        too_many_tool_call=too_many_tool_call,
    )
    return reward_parts["reward"], {
        "answer": answer,
        "format_error": format_error,
        "format_score": reward_parts["format_score"],
        "search_score": reward_parts["search_score"],
        "repeated_query_penalty": reward_parts["repeated_query_penalty"],
        "too_many_tool_call_penalty": reward_parts["too_many_tool_call_penalty"],
        "tool_call_count": tool_call_count,
        "outcome_score": outcome_score,
        "overlong_penalty": overlong_penalty,
        "outcome_reward": outcome_score + overlong_penalty,
        "truncation_penalty": truncation_penalty if truncated else 0.0,
        "context_tokens": context_token_size,
        "reward": reward_parts["reward"],
    }


def _float_metadata_value(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def _outcome_reward_from_metadata(metadata: Any) -> float | None:
    if not isinstance(metadata, dict):
        return None

    outcome_reward = _float_metadata_value(metadata, "outcome_reward")
    if outcome_reward is not None:
        return outcome_reward

    outcome_score = _float_metadata_value(metadata, "outcome_score")
    overlong_penalty = _float_metadata_value(metadata, "overlong_penalty")
    if outcome_score is None or overlong_penalty is None:
        return None
    return outcome_score + overlong_penalty


def _outcome_reward_train_metadata(metadata: Any) -> dict[str, float]:
    if not isinstance(metadata, dict):
        return {}

    out: dict[str, float] = {}
    for key in ("outcome_score", "overlong_penalty"):
        value = _float_metadata_value(metadata, key)
        if value is not None:
            out[key] = value
    outcome_reward = _outcome_reward_from_metadata(metadata)
    if outcome_reward is not None:
        out["outcome_reward"] = outcome_reward
    return out


def _is_igpo_enabled(args: Any, *, evaluation: bool) -> bool:
    return not evaluation and _arg_value(args, "advantage_estimator", "grpo") == "igpo"


def _igpo_reward_side(args: Any) -> str:
    value = str(_arg_value(args, "searcherkit_igpo_reward_side", "rollout"))
    if value not in {"actor", "rollout"}:
        raise ValueError(f"Unsupported searcherkit_igpo_reward_side={value!r}; expected 'actor' or 'rollout'")
    return value


def _answer_tokens(tokenizer: Any, answer: str) -> tuple[list[int], int, int]:
    content = f"\\boxed{{{answer}}}"
    encoded = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": content}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(encoded, dict):
        token_ids = list(encoded["input_ids"])
    else:
        token_ids = list(encoded)
    # Qwen chat template: <|im_start|>assistant\n + content + <|im_end|>\n.
    content_start = 3
    content_end = max(content_start, len(token_ids) - 2)
    return token_ids, content_start, content_end


def _score_masked_logprob(output: dict[str, Any], mask: list[bool]) -> float:
    meta = output.get("meta_info") or {}
    token_logprobs = meta.get("input_token_logprobs") or []
    values: list[float] = []
    for item, keep in zip(token_logprobs, mask, strict=False):
        if not keep:
            continue
        if item is None:
            continue
        values.append(float(item[0]))
    if not values:
        return 0.0
    return sum(values) / len(values)


async def _score_igpo_answer_logprobs(
    args: Any,
    tokenizer: Any,
    turns: list[Any],
    answer: str,
) -> list[float]:
    if not turns or not answer:
        return []

    answer_token_ids, answer_start, answer_end = _answer_tokens(tokenizer, answer)
    model_name = _arg_value(args, "searcherkit_igpo_model_name") or _arg_value(
        args,
        "searcherkit_rollout_model_name",
        "default",
    )
    url = get_model_url(args, model_name, "/generate")
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 0,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }

    logprobs: list[float] = []
    for turn in turns:
        prefix = list(turn.prompt_ids)
        full = prefix + answer_token_ids
        mask = [False] * len(full)
        # input_token_logprobs[i] scores full[i], so mark answer content tokens directly.
        start = len(prefix) + answer_start
        end = len(prefix) + answer_end
        for idx in range(start, min(end, len(mask))):
            mask[idx] = True
        output = await post(
            url,
            {
                "input_ids": full,
                "sampling_params": sampling_params,
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        )
        if not isinstance(output, dict):
            raise TypeError(f"SGLang /generate returned {type(output).__name__}, expected dict")
        logprobs.append(_score_masked_logprob(output, mask))
    return logprobs


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    idx = 0
    while idx < size and left[idx] == right[idx]:
        idx += 1
    return idx


def _turn_response_spans_in_segment(turns: list[Any], response_length: int) -> list[tuple[int, int]]:
    if not turns:
        return []

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    spans: list[tuple[int, int]] = []

    for idx, turn in enumerate(turns):
        if idx > 0:
            if turn.prompt_ids[: len(prompt_ids)] != prompt_ids:
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                spans = []
            else:
                prompt_suffix = list(turn.prompt_ids[len(prompt_ids) :])
                matched_len = _common_prefix_len(response_ids, prompt_suffix)
                if matched_len < len(response_ids):
                    response_ids = response_ids[:matched_len]
                    spans = [(start, min(end, matched_len)) for start, end in spans if start < matched_len]
                response_ids.extend(prompt_suffix[matched_len:])

        start = len(response_ids)
        response_ids.extend(turn.output_ids)
        end = min(len(response_ids), response_length)
        if end > start:
            spans.append((start, end))
        if len(response_ids) >= response_length:
            break
    return spans


def _build_igpo_token_rewards(
    *,
    response_length: int,
    spans: list[tuple[int, int]],
    answer_logprobs: list[float],
    turn_indices: list[int] | None = None,
) -> tuple[list[float], list[int], list[float]]:
    raw_rewards = [answer_logprobs[i + 1] - answer_logprobs[i] for i in range(max(0, len(answer_logprobs) - 1))]
    token_rewards = [0.0] * response_length
    reward_mask = [0] * response_length
    for idx, span in enumerate(spans):
        reward_idx = idx
        if turn_indices is not None:
            if idx >= len(turn_indices):
                break
            reward_idx = turn_indices[idx]
        if reward_idx >= len(raw_rewards):
            break
        _, end = span
        pos = end - 1
        if 0 <= pos < response_length:
            token_rewards[pos] = float(raw_rewards[reward_idx])
            reward_mask[pos] = 1
    return token_rewards, reward_mask, raw_rewards


async def _attach_igpo_metadata(
    args: Any,
    sample: Sample,
    tokenizer: Any,
    turns: list[Any],
    response_length: int,
    label: str,
) -> None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    span_source = "segment"
    spans = metadata.get("train_model_output_spans")
    turn_indices = metadata.get("train_model_output_turn_indices")
    if not spans:
        spans = metadata.get("model_output_spans")
        turn_indices = metadata.get("model_output_turn_indices")
    if not isinstance(spans, list) or not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in spans):
        spans = _turn_response_spans_in_segment(turns, response_length)
        turn_indices = None
        span_source = "reconstructed"
    spans = [(int(start), int(end)) for start, end in spans]
    if isinstance(turn_indices, list):
        turn_indices = [int(item) for item in turn_indices]
    else:
        turn_indices = None
    answer_logprobs: list[float] = []
    ig_token_rewards: list[float] = []
    ig_reward_mask: list[int] = []
    ig_turn_rewards: list[float] = []
    if _igpo_reward_side(args) == "rollout":
        answer_logprobs = await _score_igpo_answer_logprobs(args, tokenizer, turns, label)
        ig_token_rewards, ig_reward_mask, ig_turn_rewards = _build_igpo_token_rewards(
            response_length=response_length,
            spans=spans,
            answer_logprobs=answer_logprobs,
            turn_indices=turn_indices,
        )
    aligned_rewards = int(sum(ig_reward_mask))
    sample.train_metadata = {
        **(sample.train_metadata or {}),
        **_outcome_reward_train_metadata(metadata),
        "advantage_estimator": "igpo",
        "ig_turn_rewards": ig_turn_rewards,
        "ig_answer_logprobs": answer_logprobs,
        "ig_response_spans": spans,
        "ig_response_turn_indices": turn_indices,
        "ig_response_span_source": span_source,
        "ground_truth": label,
    }
    if _igpo_reward_side(args) == "rollout":
        sample.train_metadata["ig_token_rewards"] = ig_token_rewards
        sample.train_metadata["ig_reward_mask"] = ig_reward_mask
    abs_rewards = [abs(value) for value in ig_turn_rewards]
    sample.metadata = {
        **(sample.metadata or {}),
        "ig_turn_count": len(ig_turn_rewards),
        "ig_reward_sum": float(sum(ig_turn_rewards)),
        "ig_reward_abs_mean": float(sum(abs_rewards) / len(abs_rewards)) if abs_rewards else 0.0,
        "ig_reward_mask_count": aligned_rewards,
        "ig_reward_unmatched_count": max(0, len(ig_turn_rewards) - aligned_rewards),
        "ig_span_reconstructed_count": int(span_source == "reconstructed"),
    }


def _empty_sample(sample: Sample, tokenizer: Any, reward: float, metadata: dict[str, Any]) -> Sample:
    prompt = sample.prompt if isinstance(sample.prompt, str) else _question_from_sample(sample)
    sample.tokens = tokenizer.encode(prompt, add_special_tokens=False)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.rollout_id = sample.index
    sample.reward = float(reward)
    sample.status = Sample.Status.FAILED
    sample.metadata = {**(sample.metadata or {}), **metadata}
    return sample


async def generate_searcherkit(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """slime ``--custom-generate-function-path`` entry for SearcherKit."""

    state = GenerateState(args)
    agent_config = _agent_config_for(args, evaluation=evaluation)
    client = SlimeSGLangClient(
        args=args,
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        default_kwargs=agent_config.llm_client.default_kwargs,
        use_provider_tools=get_parser(agent_config.parser).uses_provider_tools,
        tool_call_parser=_arg_value(args, "searcherkit_tool_call_parser", "qwen"),
        reasoning_parser=_arg_value(args, "searcherkit_reasoning_parser", "qwen3"),
        max_context_tokens=agent_config.max_tokens,
        model_name=_arg_value(args, "searcherkit_rollout_model_name", "default"),
        session_id=sample.session_id,
    )
    agent = SearchAgentTraining(config=agent_config, llm_client=client)

    format_error = False
    context_error = False
    timeout_error = False
    tool_parser_error = False
    repeated_query = False
    too_many_tool_call = False
    history: list[Any] = []
    try:
        run_coro = agent.run(_question_from_sample(sample), session_id=sample.index)
        timeout_seconds = agent_config.run_timeout_seconds
        if timeout_seconds is None and evaluation:
            timeout_seconds = _DEFAULT_EVAL_AGENT_TIMEOUT_SECONDS
        if timeout_seconds is None:
            history = await run_coro
        else:
            history = await asyncio.wait_for(run_coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        format_error = True
        timeout_error = True
        history = agent.history
        logger.warning("SearcherKit rollout timed out after %.1fs", timeout_seconds)
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
        tool_parser_error = True
        history = agent.history
        logger.warning(repr(exc))
    except LLMContextError as exc:
        context_error = True
        history = agent.history
        logger.warning(repr(exc))

    label = _label_from_sample(sample)
    truncated = bool(context_error or client.context_truncated)
    reward, reward_metadata = _score_history(
        history=history,
        label=label,
        context_token_size=agent.context_token_size,
        max_tokens=agent.max_tokens,
        max_tokens_prompt_margin=agent.max_tokens_prompt_margin,
        truncation_penalty=float(_arg_value(args, "searcherkit_truncation_penalty", -1.0)),
        format_error=format_error,
        repeated_query=repeated_query,
        too_many_tool_call=too_many_tool_call,
        answer_pattern=_answer_pattern(args),
        truncated=truncated,
    )
    metadata = {
        **reward_metadata,
        "label": label,
        "num_turns": agent.turn,
        "context_error": context_error,
        "timeout_error": timeout_error,
        "repeated_query": repeated_query,
        "too_many_tool_call": too_many_tool_call,
        "searched_query_count": max(int(repeated_query), count_repeated_tool_queries(history)),
        "tool_parser_error_count": int(tool_parser_error),
        "too_many_tool_call_count": int(too_many_tool_call),
        "too_many_turn_count": int(getattr(agent, "max_turn_reminder_prompted", False)),
        "response_truncated_count": count_truncated_tool_responses(history),
        "too_long_seq_truncated_count": int(truncated),
        "duplicate_search_result_count": count_duplicate_tool_results(history),
        "context_limit_prompted": bool(getattr(agent, "max_token_reminder_prompted", False)),
    }
    segment = merge_turns(
        client.turns,
        metadata=metadata,
        eos_token_id=getattr(state.tokenizer, "eos_token_id", None),
        pad_token_id=getattr(state.tokenizer, "pad_token_id", None),
    )
    if segment is None or not segment.response_ids:
        empty = _empty_sample(sample, state.tokenizer, reward, metadata)
        if _is_igpo_enabled(args, evaluation=evaluation):
            empty.train_metadata = {
                **(empty.train_metadata or {}),
                **_outcome_reward_train_metadata(metadata),
                "advantage_estimator": "igpo",
                "ig_token_rewards": [],
                "ig_reward_mask": [],
                "ig_turn_rewards": [],
                "ig_answer_logprobs": [],
                "ig_response_spans": [],
                "ground_truth": label,
            }
        return empty

    status = Sample.Status.COMPLETED
    if truncated:
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
    if _is_igpo_enabled(args, evaluation=evaluation):
        for item in out:
            await _attach_igpo_metadata(
                args,
                item,
                state.tokenizer,
                client.turns,
                item.response_length,
                label,
            )
    for item in out:
        item.status = status
    return out[0] if len(out) == 1 else out


async def custom_rm(args: Any, sample: Sample | list[Sample], **kwargs: Any) -> float | list[float]:
    """Fallback slime ``--custom-rm-path`` for already-generated SearcherKit samples."""

    if isinstance(sample, list):
        return [float(await custom_rm(args, item, **kwargs)) for item in sample]

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "reward" in metadata:
        return float(metadata["reward"])
    if "outcome_score" in metadata and "overlong_penalty" in metadata:
        return float(metadata["outcome_score"]) + float(metadata["overlong_penalty"])
    label = _label_from_sample(sample)
    matches = list(_answer_pattern(args).finditer(sample.response or ""))
    answer = matches[0].group("answer").strip() if len(matches) == 1 else ""
    return f1_score(answer.lower(), label.lower()) if answer else 0.0


def _sample_reward_value(args: Any, sample: Sample | list[Sample]) -> float:
    if isinstance(sample, list):
        values = [float(item.get_reward_value(args)) for item in sample]
        return sum(values)
    return float(sample.get_reward_value(args))


def _sample_outcome_reward_value(sample: Sample | list[Sample]) -> float:
    if isinstance(sample, list):
        return sum(_sample_outcome_reward_value(item) for item in sample)

    value = _outcome_reward_from_metadata(sample.metadata)
    if value is None:
        sample_id = getattr(sample, "index", None)
        raise ValueError(
            "AReal-style outcome reward requires sample metadata with "
            f"'outcome_reward' or both 'outcome_score' and 'overlong_penalty'; sample={sample_id}"
        )
    return value


def mixed_reward_filter(args: Any, samples: list[Sample | list[Sample]], **kwargs: Any) -> DynamicFilterOutput:
    rewards = [_sample_reward_value(args, sample) for sample in samples]
    keep = bool(rewards) and min(rewards) < max(rewards)
    reason = None if keep else f"constant_reward_{round(rewards[0], 3) if rewards else 'empty'}"
    return DynamicFilterOutput(keep=keep, reason=reason)


def areal_outcome_reward_filter(
    args: Any,
    samples: list[Sample | list[Sample]],
    **kwargs: Any,
) -> DynamicFilterOutput:
    rewards = [_sample_outcome_reward_value(sample) for sample in samples]
    if not rewards:
        return DynamicFilterOutput(keep=False, reason="outcome_mean_empty")

    mean_reward = sum(rewards) / len(rewards)
    keep = 0.0 < mean_reward < 1.0
    reason = None if keep else f"outcome_mean_{round(mean_reward, 3)}"
    return DynamicFilterOutput(keep=keep, reason=reason)


def _post_process_scalar_rewards(args: Any, raw_rewards: list[float]) -> list[float]:
    if not raw_rewards:
        return []

    if (
        _arg_value(args, "advantage_estimator") in ["grpo", "igpo", "gspo", "reinforce_plus_plus_baseline"]
        and bool(_arg_value(args, "rewards_normalization", True))
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        group_size = int(_arg_value(args, "n_samples_per_prompt", 1))
        expected_size = group_size * int(_arg_value(args, "rollout_batch_size", 1))
        if rewards.shape[-1] == expected_size:
            rewards = rewards.reshape(-1, group_size)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if (
            _arg_value(args, "advantage_estimator") in ["grpo", "igpo", "gspo"]
            and bool(_arg_value(args, "grpo_std_normalization", True))
        ):
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return rewards.flatten().tolist()

    return raw_rewards


def areal_outcome_reward_post_process(
    args: Any,
    samples: list[Sample | list[Sample]],
) -> tuple[list[float], list[float]]:
    raw_rewards = [_sample_outcome_reward_value(sample) for sample in samples]
    return raw_rewards, _post_process_scalar_rewards(args, raw_rewards)


def _flatten_samples(samples: list[Sample | list[Sample]]) -> list[Sample]:
    out: list[Sample] = []
    for sample in samples:
        if isinstance(sample, list):
            out.extend(sample)
        else:
            out.append(sample)
    return out


def searcherkit_rollout_log(
    rollout_id: int,
    args: Any,
    samples: list[Sample | list[Sample]],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Add SearcherKit trajectory diagnostics and keep slime's default rollout logging."""

    flat_samples = _flatten_samples(samples)
    metrics = _searcherkit_metadata_metrics(flat_samples)
    if not metrics:
        return False

    log_dict = dict_add_prefix(metrics, "rollout/searcherkit/")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")
    logger.info("searcherkit rollout diagnostics %s: %s", rollout_id, log_dict)
    return False


def searcherkit_eval_rollout_log(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Add SearcherKit trajectory diagnostics and keep slime's default eval logging."""

    log_dict: dict[str, float] = {}
    has_searcherkit_metrics = False
    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples")
        if not samples:
            continue
        metrics = _searcherkit_metadata_metrics(samples)
        if metrics:
            has_searcherkit_metrics = True
            log_dict |= dict_add_prefix(metrics, f"eval/{dataset_name}/searcherkit/")
    if not has_searcherkit_metrics:
        return False

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    logger.info("searcherkit eval diagnostics %s: %s", rollout_id, log_dict)
    return False


def _searcherkit_metadata_metrics(samples: list[Sample]) -> dict[str, float]:
    metric_keys = (
        "format_score",
        "search_score",
        "outcome_score",
        "overlong_penalty",
        "truncation_penalty",
        "tool_call_count",
        "searched_query_count",
        "tool_parser_error_count",
        "too_many_tool_call_count",
        "too_many_turn_count",
        "response_truncated_count",
        "too_long_seq_truncated_count",
        "duplicate_search_result_count",
        "reward",
        "outcome_reward",
        "ig_turn_count",
        "ig_reward_sum",
        "ig_reward_abs_mean",
        "ig_reward_mask_count",
        "ig_reward_unmatched_count",
        "ig_span_reconstructed_count",
        "train_model_output_span_count",
        "model_output_span_count",
        "merge_prefix_drift_count",
        "merge_concat_recovered_count",
        "merge_truncated_response_tokens",
        "train_span_per_turn",
        "train_span_per_tool_call",
        "merge_concat_recovery_ratio",
    )
    series: dict[str, list[float]] = {key: [] for key in metric_keys}
    bool_keys = (
        "format_error",
        "context_error",
        "timeout_error",
        "repeated_query",
        "too_many_tool_call",
        "context_limit_prompted",
    )
    bool_series: dict[str, list[float]] = {key: [] for key in bool_keys}

    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        train_spans = metadata.get("train_model_output_spans")
        if isinstance(train_spans, list):
            metadata = {**metadata, "train_model_output_span_count": len(train_spans)}
        model_spans = metadata.get("model_output_spans")
        if isinstance(model_spans, list):
            metadata = {**metadata, "model_output_span_count": len(model_spans)}
        train_span_count = metadata.get("train_model_output_span_count")
        if isinstance(train_span_count, (int, float)) and not isinstance(train_span_count, bool):
            num_turns = metadata.get("num_turns")
            if isinstance(num_turns, (int, float)) and not isinstance(num_turns, bool) and num_turns > 0:
                metadata = {**metadata, "train_span_per_turn": float(train_span_count) / float(num_turns)}
            tool_call_count = metadata.get("tool_call_count")
            if (
                isinstance(tool_call_count, (int, float))
                and not isinstance(tool_call_count, bool)
                and tool_call_count > 0
            ):
                metadata = {**metadata, "train_span_per_tool_call": float(train_span_count) / float(tool_call_count)}
        drift_count = metadata.get("merge_prefix_drift_count")
        recovered_count = metadata.get("merge_concat_recovered_count")
        if (
            isinstance(drift_count, (int, float))
            and not isinstance(drift_count, bool)
            and drift_count > 0
            and isinstance(recovered_count, (int, float))
            and not isinstance(recovered_count, bool)
        ):
            metadata = {**metadata, "merge_concat_recovery_ratio": float(recovered_count) / float(drift_count)}
        for key in metric_keys:
            value = metadata.get(key)
            if isinstance(value, bool):
                series[key].append(float(value))
            elif isinstance(value, (int, float)):
                series[key].append(float(value))
        for key in bool_keys:
            value = metadata.get(key)
            if isinstance(value, bool):
                bool_series[key].append(float(value))

    metrics: dict[str, float] = {}
    for key, values in series.items():
        if not values:
            continue
        metrics |= dict_add_prefix(compute_statistics(values), f"{key}/")
    for key, values in bool_series.items():
        if values:
            metrics[f"{key}_ratio"] = sum(values) / len(values)
    return metrics
