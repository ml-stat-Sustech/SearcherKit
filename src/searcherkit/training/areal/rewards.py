from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import json
import re
import string
from typing import Any


DEFAULT_FORMAT_REWARD = 0.1

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

def searcherkit_reward_components(
    *,
    outcome_score: float,
    overlong_penalty: float,
    format_error: bool,
    format_reward: float = DEFAULT_FORMAT_REWARD,
) -> dict[str, float]:
    format_score = 0.0 if format_error else format_reward
    reward = (
        outcome_score
        + overlong_penalty
        + format_score
    )
    return {
        "reward": reward,
        "outcome_score": outcome_score,
        "format_score": format_score,
        "overlong_penalty": overlong_penalty
    }
