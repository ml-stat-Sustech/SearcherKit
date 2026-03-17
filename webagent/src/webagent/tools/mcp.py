"""Tool base classes used by agent-side tools.

This module provides:
- `BaseMCPTool`: fastmcp>=3 client wrapper with connection reuse, native async
  concurrency, logging, and robust error handling.
- `MCPTool`: concrete MCP tool that forwards arguments directly.

How To Use In Project
---------------------------------------
1. Async usage (recommended for backend services):

    async with GenericMCPTool(
        mcp_tool_name="search",
        endpoint="http://127.0.0.1:8301/mcp/",
    ) as tool:
        await tool.init()                 # connect + validate tool exists
        result = await tool.run(query="OpenAI")

2. Sync usage (scripts / quick tests):

    result = run(
        endpoint="http://127.0.0.1:8301/mcp/",
        tool_name="search",
        arguments={"query": "OpenAI"},
    )

Key Notes
---------
- `endpoint` selects the MCP server; `tool_name` selects the tool on that server.
- `init()` is idempotent and safe to call multiple times.
- If `run()` is called before `init()`, lazy initialization will run automatically.
- On init, `_ensure_tool_exists()` checks whether `mcp_tool_name` is registered
  on the target server, so configuration errors fail fast.

Quick Test (CLI)
----------------
python -m webagent.tools.tool --endpoint http://127.0.0.1:8100/mcp/ --tool search --args '{"query":"test"}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from contextlib import nullcontext
from collections.abc import Mapping
from typing import Any, Optional, Union, Literal

from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.client.transports import SSETransport, StreamableHttpTransport

from webagent.tools import BaseTool
from webagent.log import get_logger, get_trace_id, setup_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 异常类型
# ---------------------------------------------------------------------------

class ToolRecoverableError(RuntimeError):
    """可恢复的工具错误——不应终止整个运行流程，由上层 planner 决定重试策略。"""


class ToolFatalError(RuntimeError):
    """致命工具错误——通常意味着配置/连接不可用，应直接上抛给调用者。"""


# ---------------------------------------------------------------------------
# MCPTool：基于 fastmcp 的客户端工具基类
# ---------------------------------------------------------------------------

class BaseMCPTool(BaseTool):
    """可复用的 fastmcp 客户端工具基类。

    特性：
    - 连接可复用（persistent session），减少握手开销
    - asyncio.Semaphore 控制并发上限
    - 内置 trace logging，方便调试
    - 支持 async with 上下文管理器，自动关闭连接

    生命周期：
        tool = BaseMCPTool(endpoint="...")
        await tool.init()       # 建立连接并验证工具存在
        result = await tool.run(...) # 需要在子类中实现
        await tool.close()      # 释放连接
    或者：
        async with MCPTool(endpoint="...") as tool:
            await tool.init()
            result = await tool.run(...)
    """

    # 子类应覆盖此属性，指定要调用的 MCP 工具名称
    mcp_tool_name: str

    def __init__(
        self,
        name: str,
        description: str | None = None,
        arguments_schema: Optional[Mapping[str, Any]] = None,
        *,
        endpoint: str,
        auth_header: Optional[str] = None,
        transport: Union[Literal["sse"], Literal["streamable-http"]] = "sse",
        max_concurrency: Optional[int] = None,
        response_char_limit: Optional[int] = None,
        enable_trace_logging: bool = True,
        raise_on_fatal: bool = True,
        raise_argument_validation_error: bool = False,
    ) -> None:
        super().__init__(
            name,
            description,
            arguments_schema,
            raise_argument_validation_error=raise_argument_validation_error,
        )
        
        self.endpoint = endpoint
        self.auth_header = auth_header
        if not transport in ("sse", "streamable-http"):
            raise ValueError(f"unsupported transport: {transport!r}")
        self.transport = transport
        if max_concurrency and max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive: {max_concurrency}")
        self.max_concurrency = max_concurrency
        if response_char_limit and response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {response_char_limit}")
        self.response_char_limit = response_char_limit
        self.enable_trace_logging = enable_trace_logging
        self.raise_on_fatal = raise_on_fatal

        self._client: Client | None = None           # fastmcp.Client 实例
        self._connected = False            # 连接状态标记
        self._init_lock = asyncio.Lock()   # 防止 init() 并发重入
        self._conn_lock = asyncio.Lock()   # 保护底层连接操作
        self._semaphore = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency else nullcontext()
        # Track configured semaphore capacity separately. Do not compare with
        # semaphore._value, because _value is runtime-available permits.

    # def dump_metadata(self) -> dict[str, Any]:
    #     base = super().dump_metadata()
    #     base.update(
    #         {
    #             "mcp_tool_name": self.mcp_tool_name,
    #             "endpoint": self.settings.endpoint,
    #             "transport": self.settings.transport,
    #             "max_concurrency": self.settings.max_concurrency,
    #             "response_char_limit": self.settings.response_char_limit,
    #             "enable_trace_logging": self.settings.enable_trace_logging,
    #             "raise_on_fatal": self.settings.raise_on_fatal,
    #             "auth_header_set": bool(self.settings.auth_header),
    #         }
    #     )
    #     return base
    #     self._final_answer_generator = final_answer_generator

    # ---- async context manager 支持 ----

    async def __aenter__(self) -> "BaseMCPTool":
        """支持 `async with MCPTool(...) as tool:` 用法。"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出上下文时自动关闭连接。"""
        await self.close()

    # ---- 初始化 / 关闭 ----

    async def init(
        self,
        *,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 MCP 连接。

        接受三种配置方式（优先级：kwargs > conf > 构造函数/环境变量）：
        - 无参数：使用构造时已有的 settings
        - 显式 kwargs：覆盖指定字段
        - conf 参数：从 Hydra DictConfig 或 dict 中批量读取
        """
        async with self._init_lock:
            # 已连接则跳过，避免重复握手
            if self._connected and self._client is not None:
                return

            # 显式 kwargs 优先级最高，覆盖一切
            if "auth_header" in kwargs:
                if isinstance(kwargs["auth_header"], str):
                    self.auth_header = kwargs["auth_header"]
                else:
                    raise ValueError(f"auth_header must be str, got {kwargs['auth_header']}")
            if "transport" in kwargs:
                if isinstance(kwargs["transport"], str) and kwargs["transport"] in ["sse", "streamable-http"]:
                    self.transport = kwargs["transport"]
                else:
                    raise ValueError(f"transport must be 'sse' or 'streamable-http', got {kwargs['transport']}")
            if "response_char_limit" in kwargs:
                if isinstance(kwargs["response_char_limit"], int) and kwargs["response_char_limit"] > 0 :
                    self.response_char_limit = kwargs["response_char_limit"] 
                else:
                    raise ValueError(f"response_char_limit must be positive int, got {kwargs["response_char_limit"]}")
            if "enable_trace_logging" in kwargs:
                if isinstance(kwargs["enable_trace_logging"], bool):
                    self.enable_trace_logging = kwargs["enable_trace_logging"]
                else:
                    raise ValueError(f"enable_trace_logging must be bool, got {kwargs['enable_trace_logging']}")
            if "raise_on_fatal" in kwargs:
                if isinstance(kwargs["raise_on_fatal"], bool):
                    self.raise_on_fatal = kwargs["raise_on_fatal"]
                else:
                    raise ValueError(f"raise_on_fatal must be bool, got {kwargs['raise_on_fatal']}")

            # Rebuild semaphore only when configured capacity actually changes.
            # Avoid comparing with semaphore._value; that value fluctuates while
            # tasks are running and does not represent configured capacity.
            if "max_concurrency" in kwargs:
                if isinstance(kwargs["max_concurrency"], int) and kwargs["max_concurrency"] > 0 :
                    new_max_concurrency = kwargs["max_concurrency"] 
                    if new_max_concurrency != self.max_concurrency:
                        self.max_concurrency = new_max_concurrency
                        self._semaphore = asyncio.Semaphore(self.max_concurrency)
                else:
                    raise ValueError(f"max_concurrency must be positive int, got {kwargs["max_concurrency"]}")

            await self._connect(trace_id=trace_id)
            await self._ensure_tool_exists_and_apply_metadata(trace_id=trace_id)

    async def close(self) -> None:
        """优雅关闭 MCP 客户端连接。"""
        async with self._conn_lock:
            if self._client is None or not self._connected:
                return
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as exc:
                # 【修复】原代码 close() 中没有异常日志，关闭失败时完全静默
                logger.warning("Error while closing MCP client: %s", exc)
            finally:
                self._connected = False
                self._client = None
                
    # ---- 内部连接管理 ----

    async def _connect(self, *, trace_id: Optional[str] = None) -> None:
        """建立到 MCP 端点的连接（Streamable HTTP 或 SSE）。"""

        async with self._conn_lock:
            if self._connected and self._client is not None:
                return

            # 构造 HTTP headers（如有 auth token 则添加 Authorization）
            headers: dict[str, str] = {}
            if self.auth_header:
                headers["Authorization"] = self.auth_header

            transport = self._build_transport(headers=headers)
            self._client = Client(transport)

            # 尝试建立持久连接；若客户端支持 __aenter__ 则进入上下文
            try:
                await self._client.__aenter__()
                self._connected = True
                self._trace_log(
                    logging.INFO,
                    "Connected to MCP endpoint",
                    trace_id=trace_id,
                    endpoint=self.endpoint,
                    transport=self.transport,
                )
            except Exception as exc:
                self._connected = False
                self._client = None
                raise ToolFatalError(
                    f"Failed to connect MCP endpoint {self.endpoint}: {exc}"
                ) from exc

    def _build_transport(self, *, headers: dict[str, str] | None = None) -> Any:
        if self.transport == "streamable-http":
            transport_cls = StreamableHttpTransport
        elif self.transport == "sse":
            transport_cls = SSETransport
        else:
            raise ToolFatalError(
                f"Unsupported MCP transport '{self.transport}'. "
                'Use "http" or "sse".'
            )

        return transport_cls(
            url=self.endpoint,
            headers=headers if headers else None,
        )

    async def _ensure_tool_exists_and_apply_metadata(self, *, trace_id: Optional[str] = None) -> None:
        """验证目标工具在 MCP 服务端确实存在，否则快速失败。"""
        if not self._client:
            raise ToolFatalError("MCP client is not initialized.")
        try:
            tools = await self._client.list_tools()
        except Exception as exc:
            raise ToolFatalError(f"Failed to list MCP tools: {exc}") from exc

        for info in tools:
            if getattr(info, "name", None) == self.mcp_tool_name:
                self._apply_tool_metadata(info, trace_id=trace_id)
                self._trace_log(
                    logging.DEBUG,
                    "MCP tool resolved",
                    trace_id=trace_id,
                    tool_name=self.mcp_tool_name,
                )
                return
        raise ToolFatalError(
            f"MCP tool '{self.mcp_tool_name}' not found at endpoint {self.endpoint}."
        )

    def _apply_tool_metadata(self, info: Any, *, trace_id: Optional[str]) -> None:
        """从 MCP 的 tool info 中提取 description 和 arguments schema。"""
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

        self._trace_log(
            logging.DEBUG,
            "MCP tool metadata applied",
            trace_id=trace_id,
            description_set=self.description is not None,
            schema_set=self.inputSchema is not None,
        )

    @staticmethod
    def _coerce_schema(value: Any) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse MCP schema: {value}")
                return None
            value = parsed
        if isinstance(value, Mapping):
            return value
        return None

    # ---- 核心 MCP 调用 ----

    async def _run_mcp_tool(
        self,
        arguments: dict[str, Any],
        *,
        trace_id: Optional[str] = None,
    ) -> str:
        """底层 MCP 调用：含自动初始化、并发控制、响应长度守卫。"""
        # 惰性初始化：首次调用时自动连接
        if not self._connected or self._client is None:
            await self.init(trace_id=trace_id)

        trace = trace_id or get_trace_id() or self._new_trace_id()

        # semaphore 控制并发数不超过 max_concurrency
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
                # 超长响应守卫：截断或用 final_answer_generator 生成摘要
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

    # ---- 响应处理 ----

    async def _apply_response_length_guard(
        self,
        *,
        response_text: str,
        arguments: dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> str:
        """如果响应超过 response_char_limit，进行截断。"""
        if not self.response_char_limit or len(response_text) <= self.response_char_limit:
            return response_text

        self._trace_log(
            logging.WARNING,
            "Response exceeded configured limit; generating condensed final answer",
            trace_id=trace_id,
            length=len(response_text),
            limit=self.response_char_limit,
        )

        # 直接截断并提示
        clipped = response_text[: self.response_char_limit]
        return (
            f"{clipped}\n\n"
            "[Truncated] Response exceeded configured content limit. "
        )

    # ---- 异常分类处理 ----

    async def _handle_exception(self, exc: Exception, *, trace_id: Optional[str]) -> str:
        """根据异常消息关键词判断是可恢复错误还是致命错误。

        可恢复：网络超时、限流、临时不可用 -> 返回错误消息字符串，让上层重试
        致命：配置错误、逻辑错误 -> 根据 raise_on_fatal 决定抛出或降级
        """
        message = str(exc)
        lowered = message.lower()

        recoverable_markers = (
            "timeout", "tempor", "rate limit", "connection",
            "unavailable", "503", "502", "429",
        )
        is_recoverable = any(marker in lowered for marker in recoverable_markers)

        if is_recoverable:
            self._trace_log(
                logging.WARNING, "Recoverable MCP error",
                trace_id=trace_id, error=message,
            )
            raise ToolRecoverableError from exc

        self._trace_log(
            logging.ERROR, "Fatal MCP error",
            trace_id=trace_id, error=message,
        )
        fatal = ToolFatalError(message)
        if self.raise_on_fatal:
            raise fatal from exc
        raise ToolRecoverableError from exc

    # ---- 静态/工具方法 ----

    @staticmethod
    def _extract_text(response: Any) -> str:
        """从 fastmcp 响应对象中提取纯文本。

        支持的响应格式：
        1. response.content[*].text （fastmcp 标准格式）
        2. 纯字符串
        3. dict/Mapping -> JSON
        4. 其他 -> str()
        """
        if response is None:
            return ""

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
    def _new_trace_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _preview(data: Any, limit: int = 300) -> str:
        """生成参数预览，超长时截断（用于日志）。"""
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
        """统一的 trace 日志输出。

        当 enable_trace_logging=False 时，仅抑制 DEBUG 级别；
        WARNING/ERROR 等始终记录（保障可观测性）。
        """
        if level == logging.DEBUG and not self.enable_trace_logging:
            return
        payload = {"trace_id": trace_id or get_trace_id() or "-", **fields}
        logger.log(level, "%s | %s", message, payload)


# ---------------------------------------------------------------------------
# GenericMCPTool：开箱即用的 MCP 工具（无需子类化）
# ---------------------------------------------------------------------------

class MCPTool(BaseMCPTool):
    """通用 MCP 工具：直接将参数转发给指定的 `mcp_tool_name`。

    用法：
        tool = GenericMCPTool("search", endpoint="http://localhost:8100/mcp/")
        await tool.init()
        result = await tool.run(query="hello world")
    """

    def __init__(self, name, endpoint, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            endpoint=endpoint,
            **kwargs)
        self.mcp_tool_name = name

    async def _run(
        self,
        **kwargs,
    ) -> str:
        return await self._run_mcp_tool(kwargs)


# ---------------------------------------------------------------------------
# CLI 入口：方便直接 `python -m webagent.tools.tool` 进行快速测试
# ---------------------------------------------------------------------------

async def _async_main(
    endpoint: str,
    tool_name: str,
    arguments: dict[str, Any],
    auth_header: str = "",
) -> str:
    """异步测试入口：连接 MCP 服务并调用指定工具。"""
    tool = MCPTool(
        name=tool_name,
        endpoint=endpoint,
        auth_header=auth_header or None,
    )
    async with tool:
        await tool.init()
        result = await tool.run(**arguments)
    return result


def run(
    endpoint: str = "http://127.0.0.1:8100/mcp",
    tool_name: str = "search",
    arguments: Optional[dict[str, Any]] = None,
    auth_header: str = "",
) -> str:
    """同步测试入口——可在普通 Python 脚本或 REPL 中直接调用。

    示例：
        from webagent.tools.tool import run
        result = run(
            endpoint="http://127.0.0.1:8100/mcp",
            tool_name="search",
            arguments={"query": "hello"},
        )
        print(result)
    """
    if arguments is None:
        arguments = {}
    return asyncio.run(_async_main(endpoint, tool_name, arguments, auth_header))


def main() -> None:
    """命令行入口：解析参数后调用 run()。

    用法：
        python -m webagent.tools.tool \\
            --endpoint http://127.0.0.1:8100/mcp \\
            --tool search \\
            --args '{"query": "test"}'
    """
    parser = argparse.ArgumentParser(
        description="Quick test for MCP tools（快速测试 MCP 工具）",
    )
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8100/mcp",
        help="MCP server endpoint URL",
    )
    parser.add_argument(
        "--tool", required=True, dest="tool_name",
        help="Name of the MCP tool to call",
    )
    parser.add_argument(
        "--args", default="{}",
        help='JSON string of arguments, e.g. \'{"query": "test"}\'',
    )
    parser.add_argument(
        "--auth", default="",
        help="Authorization header value (optional)",
    )

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
        print(result)
    except (ToolFatalError, ToolRecoverableError) as exc:
        logger.error("Tool execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
