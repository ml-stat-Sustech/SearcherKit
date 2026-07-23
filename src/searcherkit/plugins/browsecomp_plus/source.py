"""BrowseComp Plus corpus reading and preprocessing."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from searcherkit.plugins.indexing import IndexDocument

logger = logging.getLogger(__name__)


def preprocess_browsecomp_plus_record(record: Mapping[str, Any]) -> IndexDocument | None:
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    title = ""
    extra_metadata: dict[str, str] = {}
    clean_text = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key, val = key.strip(), val.strip()
                    if key == "title":
                        title = val
                    else:
                        extra_metadata[key] = val
            clean_text = parts[2].strip()

    docid = record.get("docid")
    if not isinstance(docid, str) or not docid.strip():
        raise ValueError("docid missing or empty in record")
    docid = str(docid)

    url = record.get("url", "")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url missing or empty in record")

    if not title:
        title = clean_text.splitlines()[0][:120]
        logger.warning("Title not found in frontmatter, using text fallback for docid=%s", docid)

    return IndexDocument(
        id=docid,
        title=title,
        text=clean_text,
        url=url,
        links=[],
        metadata={"source": "browsecomp_plus", **extra_metadata},
    )


class BrowseCompPlusSource:
    """Iterate normalized documents from a local file or Hugging Face dataset."""

    def __init__(self, dataset_path: str | Path, *, split: str = "train") -> None:
        self.dataset_path = str(dataset_path)
        self.split = split

    def iter_documents(self, *, limit: int | None = None) -> Iterator[IndexDocument]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        count = 0
        for record in self._iter_records():
            document = preprocess_browsecomp_plus_record(record)
            if document is None:
                continue
            yield document
            count += 1
            if limit is not None and count >= limit:
                return

    def _iter_records(self) -> Iterator[Mapping[str, Any]]:
        from datasets import load_dataset

        path = Path(self.dataset_path)
        if path.exists():
            if path.suffix in (".jsonl", ".json"):
                dataset = load_dataset("json", data_files=str(path), split=self.split, streaming=True)
            elif path.suffix == ".parquet": 
                dataset = load_dataset("parquet", data_files=str(path), split=self.split, streaming=True)
            else:
                dataset = load_dataset(self.dataset_path, split=self.split, streaming=True)
        else:
            dataset = load_dataset(self.dataset_path, split=self.split, streaming=True)

        for row in dataset:
            if isinstance(row, Mapping):
                yield row
