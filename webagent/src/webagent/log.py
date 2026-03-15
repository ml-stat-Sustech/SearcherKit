from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
import os
from pathlib import Path
from threading import RLock
from typing import Any

DEFAULT_LOGGER_NAME = "webagent"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_ENV_VARS = ("WEBAGENT_LOG_LEVEL", "LOG_LEVEL")
LOG_FORMAT = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s"
RUN_LOG_FORMAT = (
    "%(asctime)s %(filename)s:%(lineno)d %(levelname)s "
    "[scope=%(scope)s run=%(run_id)s sample=%(sample_id)s trace=%(trace_id)s turn=%(turn)s] "
    "%(message)s"
)
TRACE_LOG_FORMAT = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s [turn=%(turn)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONSOLE_HANDLER_NAME = "webagent.console"
_RUN_FILE_HANDLER_NAME = "webagent.run_file"
_TRACE_ROUTER_HANDLER_NAME = "webagent.trace_router"
_DEFAULT_TRACE_TEXT_FILENAME_TEMPLATE = "{sample_id}_{trace_id}.log"
_DEFAULT_TRACE_JSON_FILENAME_TEMPLATE = "{sample_id}_{trace_id}.json"

_scope_var: ContextVar[str] = ContextVar("webagent_log_scope", default="global")
_run_id_var: ContextVar[str] = ContextVar("webagent_run_id", default="-")
_sample_id_var: ContextVar[str] = ContextVar("webagent_sample_id", default="-")
_trace_id_var: ContextVar[str] = ContextVar("webagent_trace_id", default="-")
_turn_var: ContextVar[str] = ContextVar("webagent_turn", default="-")


def _normalize_context_value(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text or "-"


def _sanitize_path_part(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    sanitized = "".join(allowed).strip("._")
    return sanitized or "unknown"


def get_log_context() -> dict[str, str]:
    return {
        "scope": _scope_var.get(),
        "run_id": _run_id_var.get(),
        "sample_id": _sample_id_var.get(),
        "trace_id": _trace_id_var.get(),
        "turn": _turn_var.get(),
    }


def get_trace_id() -> str | None:
    trace_id = _trace_id_var.get()
    return None if trace_id == "-" else trace_id


@contextmanager
def log_context(
    *,
    scope: str | None = None,
    run_id: str | int | None = None,
    sample_id: str | int | None = None,
    trace_id: str | None = None,
    turn: str | int | None = None,
):
    tokens: list[tuple[ContextVar[str], Token[str]]] = []
    updates = (
        (_scope_var, scope),
        (_run_id_var, run_id),
        (_sample_id_var, sample_id),
        (_trace_id_var, trace_id),
        (_turn_var, turn),
    )
    try:
        for variable, value in updates:
            if value is None:
                continue
            tokens.append((variable, variable.set(_normalize_context_value(value))))
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = get_log_context()
        record.scope = context["scope"]
        record.run_id = context["run_id"]
        record.sample_id = context["sample_id"]
        record.trace_id = context["trace_id"]
        record.turn = context["turn"]
        return True


class ScopeFilter(logging.Filter):
    def __init__(self, *scopes: str) -> None:
        super().__init__()
        self._scopes = set(scopes)

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "scope", "global") in self._scopes


def _deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = value


class TextTraceRouterHandler(logging.Handler):
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
        if getattr(record, "scope", "global") != "trace":
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
        except Exception:
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
            _ensure_handler_filter(handler, ContextFilter)
            self._handlers[trace_id] = handler
            return handler


class JsonTraceRouterHandler(logging.Handler):
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
        if getattr(record, "scope", "global") != "trace":
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
                        "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "filename": record.filename,
                        "lineno": record.lineno,
                        "turn": _normalize_context_value(getattr(record, "turn", "-")),
                    }
                )
                self._flush_state(state)
        except Exception:
            self.handleError(record)

    def update_trace(self, *, trace_id: str, sample_id: str | None = None, run_id: str | None = None, **fields: Any) -> None:
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
            "run": {
                "run_id": run_id,
            },
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
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _ensure_handler_filter(handler: logging.Handler, filter_cls: type[logging.Filter], *args: Any) -> None:
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
    with handler._lock:
        state = handler._get_state(
            sample_id=normalized_sample_id,
            trace_id=normalized_trace_id,
            run_id=normalized_run_id,
        )
        state["tool_interactions"].append(payload)
        handler._flush_state(state)


def configure_run_logging(
    *,
    output_dir: str | Path,
    cfg: Any = None,
) -> logging.Logger:
    base_logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    resolved_level = _resolve_log_level(cfg.get("level") if cfg else None)
    trace_cfg = cfg.get("trace") if cfg else None
    trace_enabled = bool(trace_cfg.get("enabled", True)) if trace_cfg else True
    trace_level = _resolve_log_level(trace_cfg.get("level")) if trace_cfg else resolved_level
    trace_format = str(trace_cfg.get("format", "text")).strip().lower() if trace_cfg else "text"
    trace_dir = (
        Path(trace_cfg.get("dir")) if trace_cfg and trace_cfg.get("dir") else Path(output_dir) / "traces"
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
    elif trace_format == "json" and raw_filename_template == _DEFAULT_TRACE_TEXT_FILENAME_TEMPLATE:
        filename_template = _DEFAULT_TRACE_JSON_FILENAME_TEMPLATE
    else:
        filename_template = raw_filename_template
    run_log_path = (
        Path(cfg.get("global_file"))
        if cfg and cfg.get("global_file")
        else Path(output_dir) / "run.log"
    )

    _remove_handler(base_logger, _RUN_FILE_HANDLER_NAME)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    run_handler.set_name(_RUN_FILE_HANDLER_NAME)
    run_handler.setLevel(resolved_level)
    run_handler.setFormatter(logging.Formatter(RUN_LOG_FORMAT, datefmt=DATE_FORMAT))
    _ensure_handler_filter(run_handler, ContextFilter)
    _ensure_handler_filter(run_handler, ScopeFilter, "global")
    base_logger.addHandler(run_handler)

    _remove_handler(base_logger, _TRACE_ROUTER_HANDLER_NAME)
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
        _ensure_handler_filter(trace_handler, ContextFilter)
        base_logger.addHandler(trace_handler)

    return base_logger


def _resolve_log_level(level: str | int | None = None) -> int:
    if isinstance(level, int):
        return level

    level_name = level
    if level_name is None:
        for env_var in LOG_LEVEL_ENV_VARS:
            env_value = os.getenv(env_var)
            if env_value:
                level_name = env_value
                break
    level_name = (level_name or DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    level: str | int | None = None,
) -> logging.Logger:
    resolved_level = _resolve_log_level(level)
    base_logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    base_logger.setLevel(resolved_level)
    base_logger.propagate = False

    handler = None
    for existing_handler in base_logger.handlers:
        if existing_handler.get_name() == _CONSOLE_HANDLER_NAME:
            handler = existing_handler
            break

    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_CONSOLE_HANDLER_NAME)
        base_logger.addHandler(handler)

    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    _ensure_handler_filter(handler, ContextFilter)

    logger = logging.getLogger(name)
    if name.startswith(f"{DEFAULT_LOGGER_NAME}."):
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
    else:
        logger.setLevel(resolved_level)
        logger.propagate = False
        if logger is not base_logger and not logger.handlers:
            logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    target_name = name or DEFAULT_LOGGER_NAME
    return setup_logger(target_name)


logger = get_logger()

