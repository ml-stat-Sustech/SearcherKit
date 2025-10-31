"""Unified dataset loader used across the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset as hf_load_dataset

from .gaia import is_gaia_dataset_root, load_gaia_records


def _load_local_json_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load records from a local .json or .jsonl file."""

    records: List[Dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:  # noqa: BLE001
                    raise ValueError(f"Unable to parse line in JSONL dataset: {path}") from exc
                if isinstance(entry, dict):
                    records.append(entry)
    else:
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise ValueError(f"Unable to parse JSON dataset: {path}") from exc

        if isinstance(loaded, dict):
            verified_pairs = loaded.get("verified_qa_pairs")
            if isinstance(verified_pairs, list):
                print('1111')
                for entry in verified_pairs:
                    if not isinstance(entry, dict):
                        continue
                    question = entry.get("question")
                    answer = entry.get("answer")
                    if question is None or answer is None:
                        continue
                    record = dict(entry)
                    record["question"] = question
                    record["answer"] = answer
                    records.append(record)
                if records:
                    return records

            candidate = loaded.get("data")
            if isinstance(candidate, list):
                loaded = candidate

        if isinstance(loaded, list):
            records.extend([entry for entry in loaded if isinstance(entry, dict)])

    return records


def load_dataset_records(dataset_name: str, split: str):
    """
    Load dataset entries for the given name/split.

    - Prefers GAIA-specific local loading when metadata files are detected.
    - Falls back to Hugging Face `load_dataset` for all other datasets,
      including remote identifiers and local dataset directories such as WebWalkerQA.
    - Supports direct loading of local JSON/JSONL files when `dataset_name` ends with `.json` or `.jsonl`.
    """
    dataset_path = Path(dataset_name).expanduser()
    suffix = dataset_path.suffix.lower()

    if is_gaia_dataset_root(dataset_path):
        return load_gaia_records(dataset_path, split)

    if dataset_path.exists():
        if suffix in {".json", ".jsonl"}:
            print(f"===================== Loading local datasets =====================")
            return _load_local_json_dataset(dataset_path)
        dataset_name = str(dataset_path)
    elif suffix in {".json", ".jsonl"}:
        raise FileNotFoundError(f"JSON dataset path does not exist: {dataset_path}")

    return hf_load_dataset(path=dataset_name, split=split)
