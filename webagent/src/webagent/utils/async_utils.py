from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TypeVar

T = TypeVar("T")

def ensure_async_iter(obj: Iterable[T] | AsyncIterable[T]) -> AsyncIterable[T]:
    """Return an async iterable for either sync or async iterable input."""
    if isinstance(obj, AsyncIterable):
        return obj

    async def _iter() -> AsyncIterator[T]:
        for item in obj:
            yield item

    return _iter()
