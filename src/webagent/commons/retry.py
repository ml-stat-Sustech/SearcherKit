from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

import backoff

from webagent.log import get_logger
from webagent.commons.utils import get_or_default

P = ParamSpec("P")
T = TypeVar("T")

logger = get_logger(__name__)


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

    def __post_init__(self) -> None:
        excs = self.exceptions
        if isinstance(excs, list) or isinstance(excs, set):
            excs = tuple(excs)
        elif isinstance(excs, type) and issubclass(excs, BaseException):
            excs = (excs,)
        if not isinstance(excs, tuple):
            raise TypeError("exceptions must be an exception type or tuple of exception types")
        invalid = [
            exc
            for exc in excs
            if not isinstance(exc, type) or not issubclass(exc, BaseException)
        ]
        if invalid:
            raise TypeError(f"exceptions must be exception types, got: {invalid!r}")
        self.exceptions = excs


def _build_handlers(
    *,
    op_name: str,
    log: logging.Logger | None,
) -> tuple[Callable[[dict[str, Any]], None], Callable[[dict[str, Any]], None]]:
    active_logger = log or logger

    def on_backoff(details: dict[str, Any]) -> None:
        exc = get_or_default(details, "exception", None)
        active_logger.warning(
            "Retrying op=%s tries=%s wait=%.2fs error=%r",
            op_name,
            get_or_default(details, "tries", "unknown"),
            get_or_default(details, "wait", 0.0),
            exc,
        )

    def on_giveup(details: dict[str, Any]) -> None:
        exc = get_or_default(details, "exception", None)
        active_logger.error(
            "Retry exhausted op=%s tries=%s error=%r",
            op_name,
            get_or_default(details, "tries", "unknown"),
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
    backoff_kwargs: dict[str, Any] = {
        "max_tries": policy.max_tries,
        "max_time": policy.max_time,
        "jitter": policy.jitter,
        "factor": policy.factor,
        "base": policy.base,
        "on_backoff": on_backoff,
        "on_giveup": on_giveup,
        "logger": None,
        "raise_on_giveup": True,
    }
    if policy.giveup is not None:
        backoff_kwargs["giveup"] = policy.giveup

    return backoff.on_exception(
        backoff.expo,
        policy.exceptions,
        **backoff_kwargs,
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
    backoff_kwargs: dict[str, Any] = {
        "max_tries": policy.max_tries,
        "max_time": policy.max_time,
        "jitter": policy.jitter,
        "factor": policy.factor,
        "base": policy.base,
        "on_backoff": on_backoff,
        "on_giveup": on_giveup,
        "logger": None,
        "raise_on_giveup": True,
    }
    if policy.giveup is not None:
        backoff_kwargs["giveup"] = policy.giveup

    return backoff.on_exception(
        backoff.expo,
        policy.exceptions,
        **backoff_kwargs,
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
