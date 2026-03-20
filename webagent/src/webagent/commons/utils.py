from __future__ import annotations

from typing import AsyncIterable, AsyncIterator, Iterable, TypeVar, Mapping, Any

T = TypeVar("T")

def ensure_async_iter(obj: Iterable[T] | AsyncIterable[T]) -> AsyncIterable[T]:
    """Return an async iterable for either sync or async iterable input."""
    if isinstance(obj, AsyncIterable):
        return obj

    async def _iter() -> AsyncIterator[T]:
        for item in obj:
            yield item

    return _iter()

def get_or_default(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def get_first_or_default(
    mapping: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping.get(key)
            if value is not None:
                return value
    return default

