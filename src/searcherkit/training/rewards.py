from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import json
import re
import string
from typing import Any


DEFAULT_FORMAT_REWARD = 0.1
DEFAULT_SEARCH_REWARD = 0.1
DEFAULT_NO_SEARCH_PENALTY = -0.1
DEFAULT_REPEATED_QUERY_PENALTY = -0.1
DEFAULT_TOO_MANY_TOOL_CALL_PENALTY = -0.1


def normalize_query(value: Any) -> str:
    """Normalize a search query for repeated-query detection."""

    text = str(value)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def should_accept(x: Any) -> bool:
    """DAPO dynamic filtering: keep groups with mixed rewards."""
    return 0 < x["rewards"].mean() < 1


def assign_overlong_penalty(context_len: int, max_len: int, margin: float) -> float:
    """Soft overlong penalty from DAPO."""
    if context_len <= (max_len - margin):
        return 0.0
    if context_len > max_len:
        return -1.0
    overlong = context_len - (max_len - margin)
    return float(-overlong) / margin


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def count_tool_calls(history: Iterable[Any]) -> int:
    """Count executed tool calls from a SearcherKit message history."""

    tool_response_count = 0
    assistant_tool_call_count = 0
    for message in history:
        tool_responses = getattr(message, "tool_responses", None) or []
        tool_response_count += len(tool_responses)

        tool_calls = getattr(message, "tool_calls", None) or []
        assistant_tool_call_count += len(tool_calls)
    return tool_response_count or assistant_tool_call_count


def count_repeated_tool_queries(history: Iterable[Any]) -> int:
    """Count repeated tool calls with normalized arguments."""

    seen: set[tuple[str, str]] = set()
    repeated = 0
    for message in history:
        for tool_call in getattr(message, "tool_calls", None) or []:
            key = _tool_call_key(tool_call)
            if key in seen:
                repeated += 1
            else:
                seen.add(key)
    return repeated


def _tool_call_key(tool_call: Any) -> tuple[str, str]:
    name = str(getattr(tool_call, "name", ""))
    arguments = getattr(tool_call, "arguments", {}) or {}
    if isinstance(arguments, dict) and "query" in arguments:
        return name, normalize_query(arguments["query"])
    try:
        return name, json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return name, str(arguments)


def count_truncated_tool_responses(history: Iterable[Any]) -> int:
    """Count tool responses that were truncated before reaching the model."""

    count = 0
    for message in history:
        for result in (getattr(message, "tool_responses", None) or {}).values():
            result = str(result or "")
            if "[Truncated]" in result or "...(truncated)" in result or "(truncated)..." in result:
                count += 1
    return count


def count_duplicate_tool_results(history: Iterable[Any]) -> int:
    """Count repeated search result payloads after lightweight text normalization."""

    seen: set[str] = set()
    duplicate_count = 0
    tool_names_by_id: dict[str, str] = {}
    for message in history:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if tool_call.id is not None:
                tool_names_by_id[tool_call.id] = tool_call.name
        for tool_call_id, result in (getattr(message, "tool_responses", None) or {}).items():
            if tool_names_by_id.get(tool_call_id) != "search":
                continue
            result = str(result or "")
            normalized = _normalize_tool_result(result)
            if not normalized:
                continue
            if normalized in seen:
                duplicate_count += 1
            else:
                seen.add(normalized)
    return duplicate_count


def _normalize_tool_result(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    if len(normalized) > 2000:
        normalized = normalized[:2000]
    return normalized


def searcherkit_reward_components(
    *,
    outcome_score: float,
    overlong_penalty: float,
    format_error: bool,
    tool_call_count: int,
    repeated_query: bool,
    too_many_tool_call: bool,
    truncation_penalty: float = 0.0,
    format_reward: float = DEFAULT_FORMAT_REWARD,
    search_reward: float = DEFAULT_SEARCH_REWARD,
    no_search_penalty: float = DEFAULT_NO_SEARCH_PENALTY,
    repeated_query_penalty: float = DEFAULT_REPEATED_QUERY_PENALTY,
    too_many_tool_call_penalty: float = DEFAULT_TOO_MANY_TOOL_CALL_PENALTY,
) -> dict[str, float]:
    format_score = 0.0 if format_error else format_reward
    search_score = search_reward if tool_call_count > 0 else no_search_penalty
    repeated_query_score = repeated_query_penalty if repeated_query else 0.0
    too_many_tool_call_score = too_many_tool_call_penalty if too_many_tool_call else 0.0
    reward = (
        outcome_score
        + overlong_penalty
        + truncation_penalty
        + format_score
        + search_score
        + repeated_query_score
        + too_many_tool_call_score
    )
    return {
        "reward": reward,
        "format_score": format_score,
        "search_score": search_score,
        "repeated_query_penalty": repeated_query_score,
        "too_many_tool_call_penalty": too_many_tool_call_score,
    }
