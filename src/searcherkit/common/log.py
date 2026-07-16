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

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

DEFAULT_LOGGER_NAME = "searcherkit"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_ENV_VARS = ("SEARCHERKIT_LOG_LEVEL", "LOG_LEVEL")
LOG_FORMAT = (
    "%(asctime)s %(filename)s:%(lineno)d %(levelname)s "
    "[scope=%(scope)s run=%(run_id)s sample=%(sample_id)s "
    "trace=%(trace_id)s turn=%(turn)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TRACE_LOG_FORMAT = (
    "%(asctime)s %(filename)s:%(lineno)d %(levelname)s "
    "[turn=%(turn)s] %(message)s"
)
_CONSOLE_HANDLER_NAME = "console"
_RUN_FILE_HANDLER_NAME = "run_file"
_TRACE_ROUTER_HANDLER_NAME = "trace_router"
_DEFAULT_TRACE_TEXT_FILENAME_TEMPLATE = "{sample_id}_{trace_id}.log"
_DEFAULT_TRACE_JSON_FILENAME_TEMPLATE = "{sample_id}_{trace_id}.json"

_scope: ContextVar[str] = ContextVar("scope", default="-")
_run_id: ContextVar[str] = ContextVar("run_id", default="-")
_sample_id: ContextVar[str] = ContextVar("sample_id", default="-")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_turn: ContextVar[str] = ContextVar("turn", default="-")


def _normalize_context_value(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text or "-"


def _sanitize_path_part(value: str) -> str:
    allowed = [
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value
    ]
    sanitized = "".join(allowed).strip("._")
    return sanitized or "unknown"


def get_log_context() -> dict[str, str]:
    return {
        "scope": _scope.get(),
        "run_id": _run_id.get(),
        "sample_id": _sample_id.get(),
        "trace_id": _trace_id.get(),
        "turn": _turn.get(),
    }


def get_trace_id() -> str | None:
    trace_id = _trace_id.get()
    return None if trace_id == "-" else trace_id


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
    run_id: str | int | None = None,
    sample_id: str | int | None = None,
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
        tokens.append((var, var.set(_normalize_context_value(value))))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = value


class TextTraceRouterHandler(logging.Handler):
    """Route trace-scoped records to one text file per trace identifier."""

    def __init__(
        self,
        *,
        trace_dir: str | Path,
        level: int = logging.DEBUG,
        filename_template: str = _DEFAULT_TRACE_TEXT_FILENAME_TEMPLATE,
    ) -> None:
        super().__init__(level=level)
        self.trace_dir = Path(trace_dir)
        self.filename_template = filename_template
        self._handlers: dict[str, logging.FileHandler] = {}
        self._lock = RLock()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "scope", "-") != "trace":
            return
        trace_id = _normalize_context_value(getattr(record, "trace_id", "-"))
        if trace_id == "-":
            return
        try:
            handler = self._get_handler(
                sample_id=_normalize_context_value(getattr(record, "sample_id", "-")),
                trace_id=trace_id,
            )
            handler.handle(record)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            for handler in self._handlers.values():
                handler.close()
            self._handlers.clear()
        super().close()

    def _get_handler(self, *, sample_id: str, trace_id: str) -> logging.FileHandler:
        with self._lock:
            handler = self._handlers.get(trace_id)
            if handler is not None:
                return handler

            self.trace_dir.mkdir(parents=True, exist_ok=True)
            file_name = self.filename_template.format(
                sample_id=_sanitize_path_part(sample_id),
                trace_id=_sanitize_path_part(trace_id),
            )
            handler = logging.FileHandler(self.trace_dir / file_name, encoding="utf-8")
            handler.setLevel(self.level)
            handler.setFormatter(logging.Formatter(TRACE_LOG_FORMAT, datefmt=DATE_FORMAT))
            _ensure_handler_filter(handler, _ContextFilter)
            self._handlers[trace_id] = handler
            return handler


class JsonTraceRouterHandler(logging.Handler):
    """Persist trace-scoped logs and structured metadata as one JSON document."""

    def __init__(
        self,
        *,
        trace_dir: str | Path,
        level: int = logging.DEBUG,
        filename_template: str = _DEFAULT_TRACE_JSON_FILENAME_TEMPLATE,
    ) -> None:
        super().__init__(level=level)
        self.trace_dir = Path(trace_dir)
        self.filename_template = filename_template
        self._states: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "scope", "-") != "trace":
            return
        trace_id = _normalize_context_value(getattr(record, "trace_id", "-"))
        if trace_id == "-":
            return
        try:
            with self._lock:
                state = self._get_state(
                    sample_id=_normalize_context_value(getattr(record, "sample_id", "-")),
                    trace_id=trace_id,
                    run_id=_normalize_context_value(getattr(record, "run_id", "-")),
                )
                state["events"].append(
                    {
                        "timestamp": datetime.fromtimestamp(record.created).isoformat(
                            timespec="milliseconds"
                        ),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "filename": record.filename,
                        "lineno": record.lineno,
                        "turn": _normalize_context_value(getattr(record, "turn", "-")),
                    }
                )
                self._flush_state(state)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            self.handleError(record)

    def update_trace(
        self,
        *,
        trace_id: str,
        sample_id: str | None = None,
        run_id: str | None = None,
        **fields: Any,
    ) -> None:
        normalized_trace_id = _normalize_context_value(trace_id)
        if normalized_trace_id == "-":
            return
        with self._lock:
            state = self._get_state(
                sample_id=_normalize_context_value(sample_id),
                trace_id=normalized_trace_id,
                run_id=_normalize_context_value(run_id),
            )
            _deep_merge_dict(state, fields)
            self._flush_state(state)

    def close(self) -> None:
        with self._lock:
            for state in self._states.values():
                self._flush_state(state)
            self._states.clear()
        super().close()

    def _get_state(self, *, sample_id: str, trace_id: str, run_id: str) -> dict[str, Any]:
        state = self._states.get(trace_id)
        if state is not None:
            if sample_id != "-" and state["sample"].get("sample_id") in {None, "-"}:
                state["sample"]["sample_id"] = sample_id
            if run_id != "-" and state["run"].get("run_id") in {None, "-"}:
                state["run"]["run_id"] = run_id
            return state

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        file_name = self.filename_template.format(
            sample_id=_sanitize_path_part(sample_id),
            trace_id=_sanitize_path_part(trace_id),
        )
        state = {
            "run": {"run_id": run_id},
            "sample": {
                "sample_id": sample_id,
                "trace_id": trace_id,
                "trace_file": file_name,
            },
            "execution": {},
            "stats": {},
            "tool_interactions": [],
            "events": [],
            "_path": str(self.trace_dir / file_name),
        }
        self._states[trace_id] = state
        return state

    def _flush_state(self, state: dict[str, Any]) -> None:
        path = Path(state["_path"])
        payload = {key: value for key, value in state.items() if key != "_path"}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


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


def _ensure_handler_filter(
    handler: logging.Handler,
    filter_cls: type[logging.Filter],
    *args: Any,
) -> None:
    for existing_filter in handler.filters:
        if isinstance(existing_filter, filter_cls):
            return
    handler.addFilter(filter_cls(*args))


def _remove_handler(base_logger: logging.Logger, handler_name: str) -> None:
    for handler in list(base_logger.handlers):
        if handler.get_name() != handler_name:
            continue
        base_logger.removeHandler(handler)
        handler.close()


def _get_named_handler(handler_name: str) -> logging.Handler | None:
    base_logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    for handler in base_logger.handlers:
        if handler.get_name() == handler_name:
            return handler
    return None


def update_trace_metadata(
    *,
    trace_id: str | None = None,
    sample_id: str | int | None = None,
    run_id: str | int | None = None,
    **fields: Any,
) -> None:
    """Merge structured metadata into the active JSON trace, when configured."""

    context = get_log_context()
    active_trace_id = trace_id or get_trace_id()
    if active_trace_id is None:
        return
    handler = _get_named_handler(_TRACE_ROUTER_HANDLER_NAME)
    if not isinstance(handler, JsonTraceRouterHandler):
        return
    handler.update_trace(
        trace_id=active_trace_id,
        sample_id=(
            _normalize_context_value(sample_id)
            if sample_id is not None
            else context["sample_id"]
        ),
        run_id=(
            _normalize_context_value(run_id)
            if run_id is not None
            else context["run_id"]
        ),
        **fields,
    )


def append_trace_interaction(
    interaction: dict[str, Any],
    *,
    trace_id: str | None = None,
    sample_id: str | int | None = None,
    run_id: str | int | None = None,
) -> None:
    """Append one structured Tool Interaction to the active JSON trace."""

    context = get_log_context()
    active_trace_id = trace_id or get_trace_id()
    if active_trace_id is None:
        return
    handler = _get_named_handler(_TRACE_ROUTER_HANDLER_NAME)
    if not isinstance(handler, JsonTraceRouterHandler):
        return
    normalized_trace_id = _normalize_context_value(active_trace_id)
    normalized_sample_id = (
        _normalize_context_value(sample_id)
        if sample_id is not None
        else context["sample_id"]
    )
    normalized_run_id = (
        _normalize_context_value(run_id)
        if run_id is not None
        else context["run_id"]
    )
    payload = dict(interaction)
    payload.setdefault("timestamp", datetime.now().isoformat(timespec="milliseconds"))
    payload.setdefault("turn", context["turn"])
    with handler._lock:  # noqa: SLF001
        state = handler._get_state(  # noqa: SLF001
            sample_id=normalized_sample_id,
            trace_id=normalized_trace_id,
            run_id=normalized_run_id,
        )
        state["tool_interactions"].append(payload)
        handler._flush_state(state)  # noqa: SLF001


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
    handler = _get_named_handler(_CONSOLE_HANDLER_NAME)
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_CONSOLE_HANDLER_NAME)
        base.addHandler(handler)
    handler.setLevel(base.level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    _ensure_handler_filter(handler, _ContextFilter)
    return logging.getLogger(name)


def disable_console_logging() -> None:
    """Remove the console handler so terminal UI rendering is not corrupted."""

    _remove_handler(logging.getLogger(DEFAULT_LOGGER_NAME), _CONSOLE_HANDLER_NAME)


def configure_run_logging(
    *,
    output_dir: str | Path,
    cfg: Any = None,
) -> logging.Logger:
    """Install the global run log and optional per-trace router handlers."""

    base = logging.getLogger(DEFAULT_LOGGER_NAME)
    level = _resolve_level(cfg.get("level") if cfg else None)
    trace_cfg = cfg.get("trace") if cfg else None
    trace_enabled = bool(trace_cfg.get("enabled", True)) if trace_cfg else True
    trace_level = _resolve_level(trace_cfg.get("level")) if trace_cfg else level
    trace_format = (
        str(trace_cfg.get("format", "text")).strip().lower()
        if trace_cfg
        else "text"
    )
    base.setLevel(min(level, trace_level) if trace_enabled else level)
    base.propagate = False
    trace_dir = (
        Path(trace_cfg.get("dir"))
        if trace_cfg and trace_cfg.get("dir")
        else Path(output_dir) / "traces"
    )
    raw_filename_template = (
        str(trace_cfg.get("filename_template"))
        if trace_cfg and trace_cfg.get("filename_template")
        else None
    )
    if raw_filename_template is None:
        filename_template = (
            _DEFAULT_TRACE_JSON_FILENAME_TEMPLATE
            if trace_format == "json"
            else _DEFAULT_TRACE_TEXT_FILENAME_TEMPLATE
        )
    elif (
        trace_format == "json"
        and raw_filename_template == _DEFAULT_TRACE_TEXT_FILENAME_TEMPLATE
    ):
        filename_template = _DEFAULT_TRACE_JSON_FILENAME_TEMPLATE
    else:
        filename_template = raw_filename_template

    log_path = (
        Path(cfg["global_file"])
        if cfg and cfg.get("global_file")
        else Path(output_dir) / "run.log"
    )
    _remove_handler(base, _RUN_FILE_HANDLER_NAME)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_handler = logging.FileHandler(log_path, encoding="utf-8")
    run_handler.set_name(_RUN_FILE_HANDLER_NAME)
    run_handler.setLevel(level)
    run_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    _ensure_handler_filter(run_handler, _ContextFilter)
    base.addHandler(run_handler)

    _remove_handler(base, _TRACE_ROUTER_HANDLER_NAME)
    if trace_enabled:
        if trace_format == "json":
            trace_handler: logging.Handler = JsonTraceRouterHandler(
                trace_dir=trace_dir,
                level=trace_level,
                filename_template=filename_template,
            )
        elif trace_format == "text":
            trace_handler = TextTraceRouterHandler(
                trace_dir=trace_dir,
                level=trace_level,
                filename_template=filename_template,
            )
        else:
            raise ValueError(f"Unsupported trace log format: {trace_format}")
        trace_handler.set_name(_TRACE_ROUTER_HANDLER_NAME)
        _ensure_handler_filter(trace_handler, _ContextFilter)
        base.addHandler(trace_handler)

    return base
