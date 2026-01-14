"""Helpers for loading GAIA-style datasets stored locally."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _load_records_from_metadata(metadata_path: Path) -> List[Dict[str, object]]:
    """Load a GAIA-style metadata file and attach local file paths when available."""
    records: List[Dict[str, object]] = []
    parent_dir = metadata_path.parent
    if metadata_path.suffix == ".json":
        with metadata_path.open("r", encoding="utf-8") as handle:
            try:
                loaded = json.load(handle)
            except json.JSONDecodeError:
                loaded = []
        if isinstance(loaded, dict):
            candidate = loaded.get("data")
            loaded = candidate if isinstance(candidate, list) else []
        if isinstance(loaded, list):
            for item in loaded:
                if isinstance(item, dict):
                    records.append(item)
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                records.append(record)

    for record in records:
        file_name = record.get("file_name")
        if isinstance(file_name, str) and file_name:
            attached_path = parent_dir / file_name
            record["file_path"] = str(attached_path)
        elif "file_path" not in record:
            record["file_path"] = ""
    return records


def load_gaia_records(dataset_root: Path, split: str) -> List[Dict[str, object]]:
    """Load GAIA records from a local directory or metadata file."""
    dataset_root = dataset_root.expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"GAIA dataset path does not exist: {dataset_root}")

    if dataset_root.is_file():
        if dataset_root.suffix not in {".jsonl", ".json"}:
            raise ValueError(f"Unsupported GAIA metadata file: {dataset_root}")
        return _load_records_from_metadata(dataset_root)

    if not dataset_root.is_dir():
        raise ValueError(f"Unsupported GAIA dataset location: {dataset_root}")

    metadata_paths = sorted(
        path
        for path in dataset_root.rglob("metadata.jsonl")
        if path.parent.name == split
    )
    if not metadata_paths:
        candidate = dataset_root / split / "metadata.jsonl"
        if candidate.is_file():
            metadata_paths = [candidate]

    if not metadata_paths:
        fallback = dataset_root / "metadata.jsonl"
        if fallback.is_file():
            metadata_paths = [fallback]

    if not metadata_paths:
        raise FileNotFoundError(
            f"Unable to locate GAIA metadata.jsonl under {dataset_root} for split '{split}'"
        )

    records: List[Dict[str, object]] = []
    for metadata_path in metadata_paths:
        records.extend(_load_records_from_metadata(metadata_path))
    return records


def is_gaia_dataset_root(dataset_root: Path) -> bool:
    """Heuristic to detect GAIA datasets via local metadata files."""
    dataset_root = dataset_root.expanduser()
    if not dataset_root.exists():
        return False

    if dataset_root.is_file():
        if dataset_root.suffix in {".jsonl", ".json"}:
            try:
                with dataset_root.open("r", encoding="utf-8") as handle:
                    snippet = handle.read(1024)
            except OSError:
                return False
            return '"Final answer"' in snippet or '"Annotator Metadata"' in snippet
        return False

    if dataset_root.is_dir():
        if any(dataset_root.rglob("metadata.jsonl")):
            return True

    return False
