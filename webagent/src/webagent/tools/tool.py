"""Tool base classes used by agent-side tools.

This module provides:
- `Tool`: lightweight abstract base class.
- `MCPTool`: fastmcp>=3 client wrapper with connection reuse, native async
  concurrency, logging, and robust error handling.
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import json
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport
except Exception:  # pragma: no cover - surfaced during runtime init.
    Client = None  # type: ignore[assignment]
    SSETransport = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed < minimum:
        return default
    return parsed


def _read_conf(conf: Any, key: str, default: Any = None) -> Any:
    if conf is None:
        return default
    if isinstance(conf, Mapping):
        return conf.get(key, default)
    return getattr(conf, key, default)


def _resolve_conf(conf: Any | None) -> Any | None:
    """Support both flat config and nested config under `mcp`."""
    if conf is None:
        return None
    nested = _read_conf(conf, "mcp", None)
    return nested if nested is not None else conf


class ToolRecoverableError(RuntimeError):
    """Recoverable tool failure that should not terminate the whole run."""


class ToolFatalError(RuntimeError):
    """Fatal tool failure that should be raised to caller."""


FinalAnswerGenerator = Callable[[str, str, Optional[str]], Any]


@dataclass(slots=True)
class MCPToolSettings:
    endpoint: str = os.getenv("MCP_ENDPOINT", "http://127.0.0.1:8100/mcp")
    auth_header: str = os.getenv("MCP_AUTH_HEADER", "")
    max_concurrency: int = _coerce_int(os.getenv("MCP_MAX_CONCURRENCY", 8), 8)
    response_char_limit: int = _coerce_int(
        os.getenv("MCP_RESPONSE_CHAR_LIMIT", 20000),
        20000,
        minimum=256,
    )
    enable_trace_logging: bool = _env_flag("MCP_TRACE_LOGGING", True)
    raise_on_fatal: bool = _env_flag("MCP_RAISE_ON_FATAL", True)


class Tool:
    name: str
    description: str | None
    arguments_schema: Any | None

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tool resources."""

    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool with the provided arguments."""


class MCPTool(Tool):
    """Reusable fastmcp client-side tool base.

    Notes:
    - Supports both Hydra config and plain keyword arguments.
    - Keeps a reusable client session when possible.
    - Supports native concurrent calls through asyncio + semaphore.
    """

    mcp_tool_name: str

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        auth_header: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        response_char_limit: Optional[int] = None,
        enable_trace_logging: Optional[bool] = None,
        raise_on_fatal: Optional[bool] = None,
        final_answer_generator: Optional[FinalAnswerGenerator] = None,
    ) -> None:
        self.settings = MCPToolSettings(
            endpoint=endpoint or MCPToolSettings.endpoint,
            auth_header=auth_header
            if auth_header is not None
            else MCPToolSettings.auth_header,
            max_concurrency=max_concurrency or MCPToolSettings.max_concurrency,
            response_char_limit=response_char_limit
            or MCPToolSettings.response_char_limit,
            enable_trace_logging=(
                enable_trace_logging
                if enable_trace_logging is not None
                else MCPToolSettings.enable_trace_logging
            ),
            raise_on_fatal=(
                raise_on_fatal
                if raise_on_fatal is not None
                else MCPToolSettings.raise_on_fatal
            ),
        )
        self._client: Any = None
        self._connected = False
        self._init_lock = asyncio.Lock()
        self._conn_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        self._final_answer_generator = final_answer_generator

    @classmethod
    def from_conf(cls, conf: Any | None = None, **overrides: Any) -> "MCPTool":
        """Create tool from Hydra DictConfig/dict/object, with kwargs overrides."""
        conf = _resolve_conf(conf)
        params = {
            "endpoint": _read_conf(conf, "endpoint"),
            "auth_header": _read_conf(conf, "auth_header"),
            "max_concurrency": _read_conf(conf, "max_concurrency"),
            "response_char_limit": _read_conf(conf, "response_char_limit"),
            "enable_trace_logging": _read_conf(conf, "enable_trace_logging"),
            "raise_on_fatal": _read_conf(conf, "raise_on_fatal"),
            "final_answer_generator": _read_conf(conf, "final_answer_generator"),
        }
        params.update(overrides)
        return cls(**params)

    async def init(
        self,
        endpoint: Optional[str] = None,
        *,
        conf: Any | None = None,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MCP connection.

        Accepts either:
        - no args: use env/defaults
        - explicit kwargs
        - `conf` for Hydra DictConfig or dict/object
        """
        async with self._init_lock:
            if self._connected and self._client is not None:
                return

            if endpoint:
                self.settings.endpoint = endpoint
            if conf is not None:
                conf = _resolve_conf(conf)
                self.settings.endpoint = _read_conf(
                    conf,
                    "endpoint",
                    self.settings.endpoint,
                )
                self.settings.auth_header = _read_conf(
                    conf,
                    "auth_header",
                    self.settings.auth_header,
                )
                self.settings.max_concurrency = _coerce_int(
                    _read_conf(conf, "max_concurrency", self.settings.max_concurrency),
                    self.settings.max_concurrency,
                )
                self.settings.response_char_limit = _coerce_int(
                    _read_conf(
                        conf,
                        "response_char_limit",
                        self.settings.response_char_limit,
                    ),
                    self.settings.response_char_limit,
                    minimum=256,
                )
                self.settings.enable_trace_logging = bool(
                    _read_conf(
                        conf,
                        "enable_trace_logging",
                        self.settings.enable_trace_logging,
                    )
                )
                self.settings.raise_on_fatal = bool(
                    _read_conf(conf, "raise_on_fatal", self.settings.raise_on_fatal)
                )

            # explicit kwargs override everything
            if "auth_header" in kwargs:
                self.settings.auth_header = kwargs["auth_header"]
            if "max_concurrency" in kwargs:
                self.settings.max_concurrency = _coerce_int(
                    kwargs["max_concurrency"],
                    self.settings.max_concurrency,
                )
            if "response_char_limit" in kwargs:
                self.settings.response_char_limit = _coerce_int(
                    kwargs["response_char_limit"],
                    self.settings.response_char_limit,
                    minimum=256,
                )
            if "enable_trace_logging" in kwargs:
                self.settings.enable_trace_logging = bool(kwargs["enable_trace_logging"])
            if "raise_on_fatal" in kwargs:
                self.settings.raise_on_fatal = bool(kwargs["raise_on_fatal"])
            if "final_answer_generator" in kwargs:
                self._final_answer_generator = kwargs["final_answer_generator"]

            self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)

            await self._connect(trace_id=trace_id)
            await self._ensure_tool_exists(trace_id=trace_id)

    async def close(self) -> None:
        """Gracefully close/release the MCP client session."""
        async with self._conn_lock:
            if self._client is None or not self._connected:
                return
            try:
                aexit = getattr(self._client, "__aexit__", None)
                if callable(aexit):
                    await aexit(None, None, None)
            finally:
                self._connected = False
                self._client = None

    async def run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Subclass should implement run() and call _run_mcp_tool().")

    async def run_batch(
        self,
        payloads: list[dict[str, Any]],
        *,
        trace_id: Optional[str] = None,
        fail_fast: bool = False,
    ) -> list[str]:
        """Native concurrent MCP calls with optional fail-fast behavior."""
        if not payloads:
            return []
        jobs = [
            self._run_mcp_tool(payload, trace_id=trace_id)
            for payload in payloads
        ]
        if fail_fast:
            return await asyncio.gather(*jobs)
        results = await asyncio.gather(*jobs, return_exceptions=True)
        out: list[str] = []
        for item in results:
            if isinstance(item, Exception):
                out.append(self._recoverable_message(item))
            else:
                out.append(item)
        return out

    async def _connect(self, *, trace_id: Optional[str] = None) -> None:
        if Client is None or SSETransport is None:
            raise ToolFatalError(
                "fastmcp is unavailable. Ensure dependency fastmcp>=3.0.0 is installed."
            )

        async with self._conn_lock:
            if self._connected and self._client is not None:
                return

            headers: dict[str, str] = {}
            if self.settings.auth_header:
                headers["Authorization"] = self.settings.auth_header
            transport = SSETransport(url=self.settings.endpoint, headers=headers or None)
            self._client = Client(transport)

            # Prefer persistent connection reuse. Fallback to lazy per-call context.
            try:
                aenter = getattr(self._client, "__aenter__", None)
                if callable(aenter):
                    await aenter()
                    self._connected = True
                else:
                    self._connected = True
                self._trace_log(
                    logging.INFO,
                    "Connected to MCP endpoint",
                    trace_id=trace_id,
                    endpoint=self.settings.endpoint,
                )
            except Exception as exc:
                self._connected = False
                self._client = None
                raise ToolFatalError(f"Failed to connect MCP endpoint {self.settings.endpoint}: {exc}") from exc

    async def _ensure_tool_exists(self, *, trace_id: Optional[str] = None) -> None:
        if not self._client:
            raise ToolFatalError("MCP client is not initialized.")
        try:
            tools = await self._client.list_tools()
        except Exception as exc:
            raise ToolFatalError(f"Failed to list MCP tools: {exc}") from exc

        for info in tools:
            if getattr(info, "name", None) == self.mcp_tool_name:
                self._trace_log(
                    logging.DEBUG,
                    "MCP tool resolved",
                    trace_id=trace_id,
                    tool_name=self.mcp_tool_name,
                )
                return
        raise ToolFatalError(
            f"MCP tool '{self.mcp_tool_name}' not found at endpoint {self.settings.endpoint}."
        )

    async def _run_mcp_tool(
        self,
        arguments: dict[str, Any],
        *,
        trace_id: Optional[str] = None,
    ) -> str:
        if not self._connected or self._client is None:
            await self.init(trace_id=trace_id)

        trace = trace_id or self._new_trace_id()
        async with self._semaphore:
            self._trace_log(
                logging.INFO,
                "Calling MCP tool",
                trace_id=trace,
                tool_name=self.mcp_tool_name,
                args_preview=self._preview(arguments),
            )
            try:
                response = await self._client.call_tool(self.mcp_tool_name, arguments)
                text = self._extract_text(response)
                text = await self._apply_response_length_guard(
                    response_text=text,
                    arguments=arguments,
                    trace_id=trace,
                )
                self._trace_log(
                    logging.DEBUG,
                    "MCP call completed",
                    trace_id=trace,
                    tool_name=self.mcp_tool_name,
                    length=len(text),
                )
                return text
            except ToolFatalError:
                raise
            except Exception as exc:
                return await self._handle_exception(exc, trace_id=trace)

    async def _apply_response_length_guard(
        self,
        *,
        response_text: str,
        arguments: dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> str:
        if len(response_text) <= self.settings.response_char_limit:
            return response_text

        self._trace_log(
            logging.WARNING,
            "Response exceeded configured limit; generating condensed final answer",
            trace_id=trace_id,
            length=len(response_text),
            limit=self.settings.response_char_limit,
        )

        if self._final_answer_generator is None:
            clipped = response_text[: self.settings.response_char_limit]
            return (
                f"{clipped}\n\n"
                "[Truncated] Response exceeded configured content limit. "
                "Configure `final_answer_generator` to auto-generate final answer."
            )

        maybe_awaitable = self._final_answer_generator(
            self.mcp_tool_name,
            response_text,
            json.dumps(arguments, ensure_ascii=False),
        )
        if inspect.isawaitable(maybe_awaitable):
            condensed = await maybe_awaitable
        else:
            condensed = maybe_awaitable

        condensed_text = str(condensed or "").strip()
        if not condensed_text:
            raise ToolFatalError(
                "final_answer_generator returned empty output while response exceeded limit."
            )
        return condensed_text

    async def _handle_exception(self, exc: Exception, *, trace_id: Optional[str]) -> str:
        # Connection-like issues are recoverable: return formatted message to let
        # planner continue. Logical/configuration issues are fatal by default.
        message = str(exc)
        lowered = message.lower()

        recoverable_markers = (
            "timeout",
            "tempor",
            "rate limit",
            "connection",
            "unavailable",
            "503",
            "502",
            "429",
        )
        is_recoverable = any(marker in lowered for marker in recoverable_markers)

        if is_recoverable:
            self._trace_log(
                logging.WARNING,
                "Recoverable MCP error",
                trace_id=trace_id,
                error=message,
            )
            return self._recoverable_message(exc)

        self._trace_log(
            logging.ERROR,
            "Fatal MCP error",
            trace_id=trace_id,
            error=message,
        )
        fatal = ToolFatalError(message)
        if self.settings.raise_on_fatal:
            raise fatal from exc
        return self._recoverable_message(fatal)

    @staticmethod
    def _extract_text(response: Any) -> str:
        if response is None:
            return ""

        # fastmcp style response.content[*].text
        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)

        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            return json.dumps(response, ensure_ascii=False)
        return str(response)

    @staticmethod
    def _recoverable_message(exc: Exception) -> str:
        return f"[Tool] recoverable error: {exc}"

    @staticmethod
    def _new_trace_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _preview(data: Any, limit: int = 300) -> str:
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _trace_log(
        self,
        level: int,
        message: str,
        *,
        trace_id: Optional[str] = None,
        **fields: Any,
    ) -> None:
        if level == logging.DEBUG and not self.settings.enable_trace_logging:
            return
        payload = {"trace_id": trace_id or "-", **fields}
        logger.log(level, "%s | %s", message, payload)


class GenericMCPTool(MCPTool):
    """Concrete MCP tool that forwards arguments directly to `mcp_tool_name`."""

    def __init__(self, mcp_tool_name: str, **kwargs: Any) -> None:
        self.mcp_tool_name = mcp_tool_name
        super().__init__(**kwargs)

    @classmethod
    def from_conf(
        cls,
        conf: Any | None = None,
        *,
        mcp_tool_name: Optional[str] = None,
        **overrides: Any,
    ) -> "GenericMCPTool":
        conf = _resolve_conf(conf)
        resolved_name = mcp_tool_name or _read_conf(conf, "mcp_tool_name") or _read_conf(conf, "name")
        if not resolved_name:
            raise ValueError("`mcp_tool_name` is required for GenericMCPTool.from_conf().")
        params = {
            "endpoint": _read_conf(conf, "endpoint"),
            "auth_header": _read_conf(conf, "auth_header"),
            "max_concurrency": _read_conf(conf, "max_concurrency"),
            "response_char_limit": _read_conf(conf, "response_char_limit"),
            "enable_trace_logging": _read_conf(conf, "enable_trace_logging"),
            "raise_on_fatal": _read_conf(conf, "raise_on_fatal"),
            "final_answer_generator": _read_conf(conf, "final_answer_generator"),
        }
        params.update(overrides)
        return cls(mcp_tool_name=resolved_name, **params)

    async def run(
        self,
        arguments: Optional[dict[str, Any]] = None,
        *,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Call the bound MCP tool.

        Supports both styles:
        - `run({"query": "..."})`
        - `run(query="...")`
        """
        payload = dict(arguments or {})
        if kwargs:
            payload.update(kwargs)
        return await self._run_mcp_tool(payload, trace_id=trace_id)