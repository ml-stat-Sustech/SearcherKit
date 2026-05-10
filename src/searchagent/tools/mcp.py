"""MCP-backed tools for agent-side tool calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Literal, Optional, TYPE_CHECKING, overload

from fastmcp import Client
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from fastmcp.exceptions import ToolError

from searchagent.log import get_logger, get_trace_id, setup_logger
from searchagent.errors import FatalError, RecoverableError
from searchagent.tools.base import BaseTool, ToolConfig

if TYPE_CHECKING:
    from fastmcp.client.client import CallToolResult

logger = get_logger(__name__)



class _PooledClient:
    __slots__ = ("client", "ref_count", "auth_header")

    def __init__(
        self,
        *,
        client: Client | None = None,
        ref_count: int = 0,
        auth_header: str | None = None,
    ) -> None:
        self.client = client
        self.ref_count = ref_count
        self.auth_header = auth_header


class BaseMCPTool(BaseTool):
    """Reusable fastmcp client wrapper with pooling and concurrency control."""

    _client_pool: dict[tuple[str, str, str], _PooledClient] = {}
    _pool_lock: asyncio.Lock = asyncio.Lock()

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        *,
        mcp_tool_name: str,
        endpoint: str,
        auth_header: str | None = None,
        transport: Literal["sse", "streamable-http"] = "streamable-http",
        max_concurrency: int | None = None,
        enable_trace_logging: bool = True,
        raise_on_fatal: bool = True,
        raise_argument_validation_error: bool = False,
    ) -> None: ...

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        *,
        mcp_tool_name: str | None = None,
        endpoint: str | None = None,
        auth_header: str | None = None,
        transport: Literal["sse", "streamable-http"] = "streamable-http",
        max_concurrency: int | None = None,
        enable_trace_logging: bool = True,
        raise_on_fatal: bool = True,
        raise_argument_validation_error: bool = False,
        config: ToolConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            resolved_mcp_tool_name = config.mcp_tool_name or config.name
            if not resolved_mcp_tool_name:
                raise ValueError("MCP tool config requires name or mcp_tool_name")
            if not config.endpoint:
                raise ValueError("MCP tool config requires endpoint")
            super().__init__(config=config)
            self.mcp_tool_name = resolved_mcp_tool_name
            self.endpoint = config.endpoint
            self.auth_header = config.auth_header
            self.transport = config.transport  # type: ignore[assignment]
            self.max_concurrency = config.max_concurrency
            self.enable_trace_logging = config.enable_trace_logging
            self.raise_on_fatal = config.raise_on_fatal
        else:
            if not name:
                raise ValueError("MCP tool requires name")
            if not mcp_tool_name:
                raise ValueError("MCP tool requires mcp_tool_name")
            if not endpoint:
                raise ValueError("MCP tool requires endpoint")
            super().__init__(
                name,
                description,
                inputSchema,
                raise_argument_validation_error=raise_argument_validation_error,
            )
            self.mcp_tool_name = mcp_tool_name
            self.endpoint = endpoint
            self.auth_header = auth_header
            self.transport = transport
            self.max_concurrency = max_concurrency
            self.enable_trace_logging = enable_trace_logging
            self.raise_on_fatal = raise_on_fatal

        if self.max_concurrency is not None and self.max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive: {self.max_concurrency}")

        self._client: Client | None = None
        self._connected = False
        self._init_lock = asyncio.Lock()
        self._semaphore = (
            asyncio.Semaphore(self.max_concurrency)
            if self.max_concurrency
            else nullcontext()
        )

    async def __aenter__(self) -> "BaseMCPTool":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def init(self) -> None:
        async with self._init_lock:
            if self._connected and self._client is not None:
                return
            await self._connect()
            await self._ensure_tool_exists_and_apply_metadata()

    async def close(self) -> None:
        cls = BaseMCPTool
        key = self._pool_key()
        if self._client is None:
            return

        client = self._client
        async with cls._pool_lock:
            entry = cls._client_pool.get(key)
            if entry is not None:
                entry.ref_count -= 1
                if entry.ref_count <= 0:
                    cls._client_pool.pop(key, None)
                    await client.__aexit__(None, None, None)
            self._client = None
            self._connected = False

    async def _connect(self) -> None:
        cls = BaseMCPTool
        key = self._pool_key()

        async with cls._pool_lock:
            entry = cls._client_pool.get(key)
            if entry is not None and entry.client is not None:
                entry.ref_count += 1
                self._client = entry.client
                self._connected = True
                self._trace_log(
                    logging.INFO,
                    "Reused pooled MCP client",
                    endpoint=self.endpoint,
                    transport=self.transport,
                    ref_count=entry.ref_count,
                )
                return

            headers = {"Authorization": self.auth_header} if self.auth_header else {}
            transport = self._build_transport(headers=headers)
            client = Client(transport)

            try:
                await client.__aenter__()
            except (OSError, TimeoutError, ToolError) as exc:
                raise FatalError(
                    f"Failed to connect MCP endpoint {self.endpoint}: {exc}"
                ) from exc

            cls._client_pool[key] = _PooledClient(
                client=client,
                ref_count=1,
                auth_header=self.auth_header,
            )
            self._client = client
            self._connected = True
            self._trace_log(
                logging.INFO,
                "Connected to MCP endpoint",
                endpoint=self.endpoint,
                transport=self.transport,
            )

    def _build_transport(self, *, headers: dict[str, str] | None = None) -> Any:
        if self.transport == "streamable-http":
            transport_cls = StreamableHttpTransport
        elif self.transport == "sse":
            transport_cls = SSETransport
        else:
            raise FatalError(
                f"Unsupported MCP transport '{self.transport}'. "
                'Use "streamable-http" or "sse".'
            )
        return transport_cls(url=self.endpoint, headers=headers or None)

    async def _ensure_tool_exists_and_apply_metadata(self) -> None:
        if self._client is None:
            raise FatalError("MCP client is not initialized")
        try:
            tools = await self._client.list_tools()
        except (OSError, TimeoutError, ToolError) as exc:
            raise FatalError(f"Failed to list MCP tools: {exc}") from exc

        for info in tools:
            if getattr(info, "name", None) == self.mcp_tool_name:
                self._apply_tool_metadata(info)
                return
        raise FatalError(
            f"MCP tool '{self.mcp_tool_name}' not found at endpoint {self.endpoint}"
        )

    def _apply_tool_metadata(self, info: Any) -> None:
        description = getattr(info, "description", None)
        schema = (
            getattr(info, "input_schema", None)
            or getattr(info, "inputSchema", None)
            or getattr(info, "parameters", None)
            or getattr(info, "schema", None)
        )
        if description is not None:
            self.description = str(description)
        if schema is not None:
            self.inputSchema = self._coerce_schema(schema)

    @staticmethod
    def _coerce_schema(value: Any) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                logger.warning("Failed to parse MCP schema: %s", value)
                return None
            value = parsed
        if isinstance(value, Mapping):
            return value
        return None

    async def _run_mcp_tool(self, arguments: dict[str, Any]) -> "CallToolResult":
        if not self._connected or self._client is None:
            await self.init()
        if self._client is None:
            raise FatalError("MCP client is not initialized")

        async with self._semaphore:
            self._trace_log(
                logging.INFO,
                "Calling MCP tool",
                tool_name=self.mcp_tool_name,
                args_preview=self._preview(arguments),
            )
            try:
                return await self._client.call_tool(self.mcp_tool_name, arguments)
            except ToolError as exc:
                await self._handle_tool_error(exc)
            except (OSError, TimeoutError) as exc:
                raise RecoverableError(str(exc)) from exc

        raise RecoverableError("MCP call did not return a response")

    async def _handle_tool_error(self, exc: ToolError) -> None:
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
        if any(marker in lowered for marker in recoverable_markers):
            raise RecoverableError(message) from exc
        if self.raise_on_fatal:
            raise FatalError(message) from exc
        raise RecoverableError(message) from exc

    @staticmethod
    def _normalize_auth(auth: str | None) -> str:
        return auth or ""

    def _pool_key(self) -> tuple[str, str, str]:
        return (
            self.endpoint,
            self.transport,
            self._normalize_auth(self.auth_header),
        )

    @staticmethod
    def _new_trace_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _preview(data: Any, limit: int = 300) -> str:
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _trace_log(self, level: int, message: str, **fields: Any) -> None:
        if level == logging.DEBUG and not self.enable_trace_logging:
            return
        payload = {"trace_id": get_trace_id() or "-", **fields}
        logger.log(level, "%s | %s", message, payload)


class MCPTool(BaseMCPTool):
    """Generic MCP tool that forwards keyword arguments to a server tool."""

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        endpoint: str,
        response_char_limit: int | None = None,
        mcp_tool_name: str | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        name: str | None = None,
        endpoint: str | None = None,
        response_char_limit: Optional[int] = None,
        *,
        mcp_tool_name: str | None = None,
        config: ToolConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            super().__init__(config=config)
            response_char_limit = config.response_char_limit
        else:
            if not name:
                raise ValueError("MCPTool requires name")
            if not endpoint:
                raise ValueError("MCPTool requires endpoint")
            super().__init__(
                name=name,
                endpoint=endpoint,
                mcp_tool_name=mcp_tool_name or name,
                **kwargs,
            )
        if response_char_limit is not None and response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {response_char_limit}")
        self.response_char_limit = response_char_limit

    async def _run(self, **kwargs: Any) -> str:
        text = self._extract_text(await self._run_mcp_tool(kwargs))
        trace = get_trace_id() or self._new_trace_id()
        return await self._apply_response_length_guard(
            response_text=text,
            arguments=kwargs,
            trace_id=trace,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        if response is None:
            return ""

        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts = [
                text
                for item in content
                if isinstance((text := getattr(item, "text", None)), str) and text
            ]
            if parts:
                return "\n".join(parts)

        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            return json.dumps(response, ensure_ascii=False)
        return str(response)

    async def _apply_response_length_guard(
        self,
        *,
        response_text: str,
        arguments: dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> str:
        if not self.response_char_limit or len(response_text) <= self.response_char_limit:
            return response_text

        self._trace_log(
            logging.WARNING,
            "Response exceeded configured limit; truncating",
            trace_id=trace_id,
            length=len(response_text),
            limit=self.response_char_limit,
        )
        clipped = response_text[: self.response_char_limit]
        return f"{clipped}\n\n[Truncated] Response exceeded configured content limit."


async def _async_main(
    endpoint: str,
    tool_name: str,
    arguments: dict[str, Any],
    auth_header: str = "",
) -> str:
    tool = MCPTool(
        name=tool_name,
        endpoint=endpoint,
        auth_header=auth_header or None,
    )
    async with tool:
        await tool.init()
        return await tool.run(**arguments)


def run(
    endpoint: str = "http://127.0.0.1:8100/mcp",
    tool_name: str = "search",
    arguments: Optional[dict[str, Any]] = None,
    auth_header: str = "",
) -> str:
    return asyncio.run(
        _async_main(endpoint, tool_name, arguments or {}, auth_header)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test for MCP tools")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8100/mcp")
    parser.add_argument("--tool", required=True, dest="tool_name")
    parser.add_argument("--args", default="{}")
    parser.add_argument("--auth", default="")
    args = parser.parse_args()

    try:
        arguments = json.loads(args.args)
    except json.JSONDecodeError as exc:
        setup_logger()
        logger.error("Invalid JSON for --args: %s", exc)
        sys.exit(1)

    setup_logger(level="DEBUG")
    try:
        result = run(
            endpoint=args.endpoint,
            tool_name=args.tool_name,
            arguments=arguments,
            auth_header=args.auth,
        )
    except (FatalError, RecoverableError) as exc:
        logger.error("Tool execution failed: %s", exc)
        sys.exit(1)
    print(result)


if __name__ == "__main__":
    main()
