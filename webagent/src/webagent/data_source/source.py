from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any


DataItem = tuple[str, dict[str, Any] | None, Any | None]


class DataSource(abc.ABC):
    """Base data source interface yielding (input, extra, answer) tuples."""

    @abc.abstractmethod
    def yield_inputs(self) -> Iterator[DataItem]:
        """Yield (input, extra, answer) tuples matching Agent.run inputs."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[DataItem]:
        return self.yield_inputs()
