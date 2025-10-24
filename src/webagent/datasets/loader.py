"""Unified dataset loader used across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_dataset as hf_load_dataset

from .gaia import is_gaia_dataset_root, load_gaia_records


def load_dataset_records(dataset_name: str, split: str):
    """
    Load dataset entries for the given name/split.

    - Prefers GAIA-specific local loading when metadata files are detected.
    - Falls back to Hugging Face `load_dataset` for all other datasets,
      including remote identifiers and local dataset directories such as WebWalkerQA.
    """
    dataset_path = Path(dataset_name).expanduser()
    if is_gaia_dataset_root(dataset_path):
        return load_gaia_records(dataset_path, split)

    if dataset_path.exists():
        dataset_name = str(dataset_path)

    return hf_load_dataset(path=dataset_name, split=split)
