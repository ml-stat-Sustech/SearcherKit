"""
Logger.
  - setup_logger / configure_run_logging: install handlers
  - log_context: set context vars for nested logging
  - get_logger: return a logger; call .info/.warning/.error/... directly
Downstream callers use stdlib logger methods. No special log APIs.

Timer.
  - LogTiming: dict-based timing tracker, ``with timer("name"):`` records
    elapsed, ``timer.to_dict()`` exports aggregated stats.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

DEFAULT_LOGGER_NAME = "searchagent"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_ENV_VARS = ("SearchAgent_LOG_LEVEL", "LOG_LEVEL")
LOG_FORMAT = (
    "%(asctime)s %(filename)s:%(lineno)d %(levelname)s "
    "[scope=%(scope)s run=%(run_id)s sample=%(sample_id)s "
    "trace=%(trace_id)s turn=%(turn)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_scope: ContextVar[str] = ContextVar("scope", default="-")
_run_id: ContextVar[str] = ContextVar("run_id", default="-")
_sample_id: ContextVar[str] = ContextVar("sample_id", default="-")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_turn: ContextVar[str] = ContextVar("turn", default="-")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.scope = _scope.get()
        record.run_id = _run_id.get()
        record.sample_id = _sample_id.get()
        record.trace_id = _trace_id.get()
        record.turn = _turn.get()
        return True


@contextmanager
def log_context(
    *,
    scope: str | None = None,
    run_id: str | None = None,
    sample_id: str | None = None,
    trace_id: str | None = None,
    turn: str | int | None = None,
) -> Iterator[None]:
    """Push logging context vars for the duration of the with-block."""
    tokens: list[tuple[ContextVar[str], Token[str]]] = []
    for var, value in (
        (_scope, scope),
        (_run_id, run_id),
        (_sample_id, sample_id),
        (_trace_id, trace_id),
        (_turn, turn),
    ):
        if value is None:
            continue
        tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class _TimingCtx:
    """Context manager yielded by LogTiming.__call__."""

    def __init__(self, owner: "LogTiming", name: str) -> None:
        self._owner = owner
        self._name = name
        self._start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "_TimingCtx":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None:
        self._done(exc_type)
        return None

    async def __aenter__(self) -> "_TimingCtx":
        self._start = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None:
        self._done(exc_type)
        return None

    def _done(self, exc_type: type[BaseException] | None) -> None:
        if self._start is None:
            return
        self.elapsed = time.perf_counter() - self._start
        records = self._owner._records  # noqa: SLF001
        records.setdefault(self._name, []).append(self.elapsed)
        if not self._owner._log:  # noqa: SLF001
            return
        logger = self._owner._logger  # noqa: SLF001
        level = self._owner._level  # noqa: SLF001
        if exc_type is None:
            logger.log(level, "%s took %.3fs", self._name, self.elapsed)
        else:
            logger.log(
                level,
                "%s failed after %.3fs (%s)",
                self._name,
                self.elapsed,
                exc_type.__name__,
            )


class LogTiming:
    """Global timer that records elapsed time for named blocks into a dict.

    Design::

        timer = LogTiming()

        with timer("llm_call"):
            result = await llm_call()
        # timer._records["llm_call"] == [2.31, ...]

        timer.to_dict()
        # {"llm_call": {"count": 1, "total": 2.31, "avg": 2.31, "times": [2.31]}}

    ``LogTiming`` stores aggregated timing data and supports both sync and
    async context-manager usage via the ``__call__`` pattern.  Set ``log=True``
    (the default) to emit an INFO-level log message after every block.
    """

    def __init__(
        self,
        *,
        log: bool = True,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        self._records: dict[str, list[float]] = {}
        self._log = log
        self._logger = logger or logging.getLogger(DEFAULT_LOGGER_NAME)
        self._level = level

    def __call__(self, name: str) -> _TimingCtx:
        """Return a timing context manager for a block named *name*."""
        return _TimingCtx(self, name)
    
    def to_dict(self) -> dict[str, Any]:
        """Export all timing records as a dict suitable for JSON serialization."""
        result: dict[str, Any] = {}
        for name, times in self._records.items():
            total = sum(times)
            result[name] = {
                "count": len(times),
                "total": round(total, 3),
                "avg": round(total / len(times), 3) if times else 0.0,
                "times": [round(t, 3) for t in times],
            }
        return result


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if level is None:
        for name in LOG_LEVEL_ENV_VARS:
            v = os.getenv(name)
            if v:
                level = v
                break
    return getattr(logging, (level or DEFAULT_LOG_LEVEL).upper(), logging.INFO)


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Return a logger for *name*; callers use stdlib .info/.warning/.error."""
    return logging.getLogger(name)


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    level: str | int | None = None,
) -> logging.Logger:
    """Install a console handler on the root logger; return a child logger."""
    base = logging.getLogger(DEFAULT_LOGGER_NAME)
    base.setLevel(_resolve_level(level))
    base.propagate = False
    if not any(h.get_name() == "console" for h in base.handlers):
        h = logging.StreamHandler()
        h.set_name("console")
        h.setLevel(base.level)
        h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        h.addFilter(_ContextFilter())
        base.addHandler(h)
    return logging.getLogger(name)


def configure_run_logging(
    *,
    output_dir: str | Path,
    cfg: dict | None = None,
) -> logging.Logger:
    """Install a run.log file handler on the root logger."""
    base = logging.getLogger(DEFAULT_LOGGER_NAME)
    level = _resolve_level(cfg.get("level") if cfg else None)
    for h in list(base.handlers):
        if h.get_name() == "run_file":
            base.removeHandler(h)
            h.close()
    log_path = (
        Path(cfg["global_file"])
        if cfg and cfg.get("global_file")
        else Path(output_dir) / "run.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(log_path, encoding="utf-8")
    h.set_name("run_file")
    h.setLevel(level)
    h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    h.addFilter(_ContextFilter())
    base.addHandler(h)
    return base
