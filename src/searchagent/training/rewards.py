from __future__ import annotations

from collections import Counter
from typing import Any


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
