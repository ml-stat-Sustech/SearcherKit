from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, Literal, overload


DataItem = tuple[str, dict[str, Any] | None, Any | None]


class BaseLoader(abc.ABC):
    """Base data source interface yielding (input, extra, answer) tuples."""
    max_items: int | None = None

    @overload
    def __init__(self, *, max_items: int | None = None, **kwargs: Any) -> None: ...

    @overload
    def __init__(self, *, config: "DataConfig") -> None: ...

    def __init__(self, *, max_items: int | None = None, config: "DataConfig | None" = None, **kwargs: Any) -> None:
        if config is not None:
            self.max_items = config.max_items
        else:
            self.max_items = max_items

    @abc.abstractmethod
    def yield_inputs(self) -> Iterator[DataItem]:
        """Yield (input, extra, answer) tuples matching Agent.run inputs."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[DataItem]:
        return self.yield_inputs()
    
DataFormat = Literal["jsonl", "parquet"]

@dataclass
class DataConfig:
    target: str | None = "pkg://searchagent.common.dataloader:GenericDataLoader"
    source: str = ""
    fmt: str = "jsonl"
    input_key: str = "prompt"
    answer_key: str = "answer"
    max_items: int | None = None

class GenericDataLoader(BaseLoader):
    """Generic data source for jsonl/parquet datasets."""
    @overload
    def __init__(self, *, config: DataConfig) -> None:
        ...

    @overload
    def __init__(self, source: str, fmt: DataFormat, input_key: str = "prompt", answer_key: str = "answer", max_items: int | None = None) -> None:
        ...

    def __init__(self, 
                 source: str | None = None, 
                 fmt: DataFormat | None = None, 
                 input_key: str | None = "prompt", 
                 answer_key: str | None = "answer", 
                 max_items: int | None = None,
                 *,
                 config: DataConfig | None = None, ) -> None:
        if config is not None:
            super().__init__(config=config)
            self.source = config.source
            self.fmt = config.fmt
            self.input_key = config.input_key
            self.answer_key = config.answer_key
            return
        super().__init__(max_items=max_items)
        self.source = source
        self.fmt = fmt
        self.input_key = input_key
        self.answer_key = answer_key

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

