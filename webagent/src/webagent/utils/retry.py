from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

import backoff

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetryPolicy:
    """Retry settings shared by function and context-based APIs."""

    max_tries: int = 3
    max_time: float | None = None
    exceptions: tuple[type[Exception], ...] = (Exception,)
    giveup: Callable[[Exception], bool] | None = None
    jitter: Callable[[float], float] | None = backoff.full_jitter
    factor: float = 1.0
    base: float = 2.0


def _build_handlers(
    *,
    op_name: str,
    log: logging.Logger | None,
) -> tuple[Callable[[dict[str, Any]], None], Callable[[dict[str, Any]], None]]:
    active_logger = log or logger

    def on_backoff(details: dict[str, Any]) -> None:
        exc = details.get("exception")
        active_logger.warning(
            "Retrying op=%s tries=%s wait=%.2fs error=%r",
            op_name,
            details.get("tries"),
            details.get("wait", 0.0),
            exc,
        )

    def on_giveup(details: dict[str, Any]) -> None:
        exc = details.get("exception")
        active_logger.error(
            "Retry exhausted op=%s tries=%s error=%r",
            op_name,
            details.get("tries"),
            exc,
        )

    return on_backoff, on_giveup


def wrap_async(
    func: Callable[P, Awaitable[T]],
    *,
    policy: RetryPolicy,
    op_name: str | None = None,
    log: logging.Logger | None = None,
) -> Callable[P, Awaitable[T]]:
    """Return an async callable wrapped by backoff retry policy."""

    operation = op_name or getattr(func, "__name__", "async_op")
    on_backoff, on_giveup = _build_handlers(op_name=operation, log=log)
    return backoff.on_exception(
        backoff.expo,
        policy.exceptions,
        max_tries=policy.max_tries,
        max_time=policy.max_time,
        giveup=policy.giveup,
        jitter=policy.jitter,
        factor=policy.factor,
        base=policy.base,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        logger=None,
        raise_on_giveup=True,
    )(func)


def wrap_sync(
    func: Callable[P, T],
    *,
    policy: RetryPolicy,
    op_name: str | None = None,
    log: logging.Logger | None = None,
) -> Callable[P, T]:
    """Return a sync callable wrapped by backoff retry policy."""

    operation = op_name or getattr(func, "__name__", "sync_op")
    on_backoff, on_giveup = _build_handlers(op_name=operation, log=log)
    return backoff.on_exception(
        backoff.expo,
        policy.exceptions,
        max_tries=policy.max_tries,
        max_time=policy.max_time,
        giveup=policy.giveup,
        jitter=policy.jitter,
        factor=policy.factor,
        base=policy.base,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        logger=None,
        raise_on_giveup=True,
    )(func)


async def retry_async(
    func: Callable[P, Awaitable[T]],
    *args: P.args,
    policy: RetryPolicy,
    op_name: str | None = None,
    log: logging.Logger | None = None,
    **kwargs: P.kwargs,
) -> T:
    """Function-style async retry helper."""

    wrapped = wrap_async(func, policy=policy, op_name=op_name, log=log)
    return await wrapped(*args, **kwargs)


def retry_sync(
    func: Callable[P, T],
    *args: P.args,
    policy: RetryPolicy,
    op_name: str | None = None,
    log: logging.Logger | None = None,
    **kwargs: P.kwargs,
) -> T:
    """Function-style sync retry helper."""

    wrapped = wrap_sync(func, policy=policy, op_name=op_name, log=log)
    return wrapped(*args, **kwargs)
