from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from .source import DataSource, DataItem

DataFormat = Literal["jsonl", "parquet"]


@dataclass(slots=True)
class GenericDataSource(DataSource):
    """Generic data source for jsonl/parquet datasets."""

    source: str
    fmt: DataFormat
    input_key: str = "prompt"
    answer_key: str = "answer"
    max_items: int | None = None

    def yield_inputs(self) -> Iterator[DataItem]:
        if self.fmt == "jsonl":
            iterator = self._yield_jsonl()
        elif self.fmt == "parquet":
            iterator = self._yield_parquet()
        else:
            raise ValueError(f"unsupported fmt: {self.fmt!r}")

        if self.max_items is None:
            yield from iterator
            return
        if self.max_items <= 0:
            return # TODO log warn/error?
        for idx, item in enumerate(iterator):
            if idx >= self.max_items:
                break
            yield item

    def _yield_jsonl(self) -> Iterator[DataItem]:
        with open(self.source, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid json at line {line_no} in {self.source}") from exc
                yield self._extract_record(record)

    def _yield_parquet(self) -> Iterator[DataItem]:
        try:
            import pyarrow.parquet as pq # TODO dependency group
        except ImportError as exc:
            raise ImportError("parquet loading requires pyarrow") from exc

        table = pq.read_table(self.source)
        for record in table.to_pylist():
            yield self._extract_record(record)

    def _extract_record(self, record: dict[str, Any]) -> DataItem:
        if self.input_key not in record:
            raise KeyError(f"missing input_key {self.input_key!r} in record")
        return record[self.input_key], None, record.get(self.answer_key)
