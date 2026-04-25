"""BrowseComp Plus corpus reading and preprocessing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from searchagent.plugins.indexing import IndexDocument


TITLE_FIELDS = ("title", "page_title", "name")
TEXT_FIELDS = ("text", "contents", "content", "body", "page_content", "document")
ID_FIELDS = ("id", "_id", "docid", "doc_id", "document_id", "url")
URL_FIELDS = ("url", "source_url", "link")


def _first_str(record: Mapping[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def preprocess_browsecomp_plus_record(record: Mapping[str, Any]) -> IndexDocument | None:
    text = _first_str(record, TEXT_FIELDS)
    if not text:
        return None
    title = _first_str(record, TITLE_FIELDS) or text.splitlines()[0][:120] or "BrowseComp Plus document"
    document_id = _first_str(record, ID_FIELDS) or title
    url = _first_str(record, URL_FIELDS) or f"browsecomp-plus://{document_id}"

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {*TITLE_FIELDS, *TEXT_FIELDS, *ID_FIELDS, *URL_FIELDS, "links"}
    }
    links = record.get("links", [])
    if not isinstance(links, list):
        links = []

    return IndexDocument(
        id=str(document_id),
        title=title,
        text=text,
        url=url,
        links=[link for link in links if isinstance(link, Mapping)],
        metadata={"source": "browsecomp_plus", **metadata},
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
        path = Path(self.dataset_path)
        if path.exists():
            yield from self._iter_local_records(path)
            return

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Loading BrowseComp Plus from Hugging Face requires the 'datasets' package."
            ) from exc
        dataset = load_dataset(self.dataset_path, split=self.split)
        for row in dataset:
            if isinstance(row, Mapping):
                yield row

    def _iter_local_records(self, path: Path) -> Iterator[Mapping[str, Any]]:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if isinstance(record, Mapping):
                        yield record
            return

        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                for record in payload:
                    if isinstance(record, Mapping):
                        yield record
            elif isinstance(payload, Mapping):
                data = payload.get("data", payload.get("documents", []))
                if isinstance(data, list):
                    for record in data:
                        if isinstance(record, Mapping):
                            yield record
            return

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Loading local BrowseComp Plus parquet/arrow files requires the 'datasets' package."
            ) from exc
        dataset = load_dataset("parquet", data_files=str(path), split=self.split)
        for row in dataset:
            if isinstance(row, Mapping):
                yield row
