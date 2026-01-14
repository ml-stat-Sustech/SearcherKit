"""Helpers for loading WebWalkerQA datasets and ground truth."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping

from .loader import load_dataset_records

DEFAULT_WEBWALKER_DATASET = "/mnt/sharedata/ssd_large/common/datasets/WebWalkerQA"


def _iter_records(dataset: Iterable[Mapping[str, Any]]):
    if hasattr(dataset, "__iter__"):
        for item in dataset:
            if isinstance(item, Mapping):
                yield item
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Mapping):
                # Some datasets can yield (index, record) pairs.
                yield item[1]
    else:
        raise TypeError(f"Unsupported dataset iterable: {type(dataset)!r}")


def _extract_field(record: Mapping[str, Any], primary: str, fallbacks: tuple[str, ...]) -> Any:
    if primary in record and record[primary]:
        return record[primary]
    for key in fallbacks:
        value = record.get(key)
        if value:
            return value
    return None


@lru_cache(maxsize=1)
def load_webwalker_ground_truth(
    dataset_name: str | None = None,
    split: str = "main",
) -> Dict[str, Dict[str, Any]]:
    """Load WebWalkerQA ground truth answers with optional dataset overrides."""
    dataset_name = dataset_name or os.environ.get("WEBWALKER_DATASET", DEFAULT_WEBWALKER_DATASET)
    dataset = load_dataset_records(dataset_name, split)

    info_lookup: Dict[str, Dict[str, Any]] = {}
    question_keys = ("question", "Question")
    answer_keys = ("answer", "Answer", "final_answer", "Final answer")
    info_keys = ("info", "Info", "metadata", "Metadata")

    for record in _iter_records(dataset):
        question = _extract_field(record, question_keys[0], question_keys[1:])
        answer = _extract_field(record, answer_keys[0], answer_keys[1:])
        if not question or not answer:
            continue
        info_value = _extract_field(record, info_keys[0], info_keys[1:])
        info_lookup[str(question)] = {
            "answer": answer,
            "info": info_value,
        }
    return info_lookup
