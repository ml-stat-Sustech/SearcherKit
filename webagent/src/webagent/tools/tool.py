"""Tool base classes used by agent-side tools.

This module provides:
- `Tool`: lightweight abstract base class for all tools.
- `MCPTool`: fastmcp>=3 client wrapper with connection reuse, native async
  concurrency, logging, and robust error handling.
- `GenericMCPTool`: concrete MCP tool that forwards arguments directly.

How To Use In Project (for maintainers)
---------------------------------------
1. Async usage (recommended for backend services):

    async with GenericMCPTool(
        mcp_tool_name="search",
        endpoint="http://127.0.0.1:8301/mcp",
    ) as tool:
        await tool.init()                 # connect + validate tool exists
        result = await tool.run(query="OpenAI")

2. Sync usage (scripts / quick tests):

    result = run(
        endpoint="http://127.0.0.1:8301/mcp",
        tool_name="search",
        arguments={"query": "OpenAI"},
    )

3. Optional batch usage:

    results = await tool.run_batch([
        {"query": "OpenAI"},
        {"query": "OpenAI history"},
    ])

Key Notes
---------
- `endpoint` selects the MCP server; `tool_name` selects the tool on that server.
- `init()` is idempotent and safe to call multiple times.
- If `run()` is called before `init()`, lazy initialization will run automatically.
- On init, `_ensure_tool_exists()` checks whether `mcp_tool_name` is registered
  on the target server, so configuration errors fail fast.

Quick Test (CLI)
----------------
python -m webagent.tools.tool --endpoint http://127.0.0.1:8100/mcp --tool search --args '{"query":"test"}'
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, ValidationError

from webagent.log import get_logger, setup_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 工具函数：从环境变量或配置对象中安全地读取参数
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔 flag，支持 1/true/yes/on（不区分大小写）。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int, *, minimum: int = 1) -> int:
    """将 value 安全地转换为 >= minimum 的整数，失败则返回 default。"""
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed < minimum:
        return default
    return parsed


def _read_conf(conf: Any, key: str, default: Any = None) -> Any:
    """从 dict/Mapping 或任意对象中读取一个配置项。"""
    if conf is None:
        return default
    if isinstance(conf, Mapping):
        value = conf.get(key, default)
        return default if value is None else value
    value = getattr(conf, key, default)
    return default if value is None else value


def _resolve_conf(conf: Any | None) -> Any | None:
    """兼容平坦配置和嵌套在 `mcp` 键下的配置结构。"""
    if conf is None:
        return None
    nested = _read_conf(conf, "mcp", None)
    return nested if nested is not None else conf


# ---------------------------------------------------------------------------
# 异常类型
# ---------------------------------------------------------------------------

class ToolRecoverableError(RuntimeError):
    """可恢复的工具错误——不应终止整个运行流程，由上层 planner 决定重试策略。"""


class ToolFatalError(RuntimeError):
    """致命工具错误——通常意味着配置/连接不可用，应直接上抛给调用者。"""


# Final answer generator 的类型别名：
#   (tool_name, full_response_text, arguments_json) -> condensed_text
FinalAnswerGenerator = Callable[[str, str, Optional[str]], Any]


# ---------------------------------------------------------------------------
# MCP 工具配置（数据类，所有字段均有环境变量级别的默认值）
# ---------------------------------------------------------------------------

@dataclass
class MCPToolSettings:
    """
    MCP 连接和行为参数。所有字段均有环境变量级别的默认值。
    """
    endpoint: str = os.getenv("MCP_ENDPOINT", "http://127.0.0.1:8100/mcp")
    auth_header: str = os.getenv("MCP_AUTH_HEADER", "")
    transport: str = os.getenv("MCP_TRANSPORT", "auto")
    max_concurrency: int = _coerce_int(os.getenv("MCP_MAX_CONCURRENCY", 8), 8)
    response_char_limit: int = _coerce_int(
        os.getenv("MCP_RESPONSE_CHAR_LIMIT", 20000),
        20000,
        minimum=256,
    )
    enable_trace_logging: bool = _env_flag("MCP_TRACE_LOGGING", True)
    raise_on_fatal: bool = _env_flag("MCP_RAISE_ON_FATAL", True)


# ---------------------------------------------------------------------------
# Tool 抽象基类
# ---------------------------------------------------------------------------

class Tool(abc.ABC):
    name: str
    description: str | None
    arguments_schema: type[BaseModel] | None
    raise_argument_validation_error: bool

    def __init__(self, *, raise_argument_validation_error: bool = False) -> None:
        self.raise_argument_validation_error = raise_argument_validation_error

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tool resources."""

    async def run(self, **kwargs: Any) -> str:
        """Execute the tool with the provided arguments."""
        if self.arguments_schema is None:
            return await self._run(kwargs)
        try:
            model = self.arguments_schema.model_validate(kwargs)
        except ValidationError as exc:
            logger.warning(
                "Tool %s arguments validation failed: %s",
                self.name,
                exc,
            )
            if self.raise_argument_validation_error:
                raise
            return f"[Tool] invalid arguments for {self.name}: {exc}"
        return await self._run(model.model_dump())

    @abc.abstractmethod
    async def _run(self, arguments: dict[str, Any]) -> str:
        """Subclasses implement actual tool execution."""
        raise NotImplementedError

    def dump_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata dict for logging."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments_schema": (
                self.arguments_schema.model_json_schema()
                if self.arguments_schema is not None
                else None
            ),
            "type": self.__class__.__name__,
        }
    
    @staticmethod
    def to_openai_tool(
        name: str,
        description: str | None = None,
        arguments_schema: type[BaseModel] | None = None,
    ) -> Mapping[str, Any]:
        description = description or ""
        if arguments_schema is None:
            parameters: Mapping[str, Any] = {}
        else:
            parameters = arguments_schema.model_json_schema()
        if not description:
            logger.warning("Tool %s has no description", name)
        if not parameters:
            logger.warning("Tool %s has no arguments schema", name)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
    
    def as_openai_tool(self) -> Mapping[str, Any]:
        return self.to_openai_tool(self.name, self.description, self.arguments_schema)



# ---------------------------------------------------------------------------
# MCPTool：基于 fastmcp 的客户端工具基类
# ---------------------------------------------------------------------------

class MCPTool(Tool):
    """可复用的 fastmcp 客户端工具基类。

    特性：
    - 支持 Hydra DictConfig / plain dict / keyword 三种配置方式
    - 连接可复用（persistent session），减少握手开销
    - asyncio.Semaphore 控制并发上限
    - 内置 trace logging，方便调试
    - 支持 async with 上下文管理器，自动关闭连接

    生命周期：
        tool = MCPTool(endpoint="...")
        await tool.init()       # 建立连接并验证工具存在
        result = await tool.run(...)
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
        *,
        endpoint: Optional[str] = None,
        auth_header: Optional[str] = None,
        transport: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        response_char_limit: Optional[int] = None,
        enable_trace_logging: Optional[bool] = None,
        raise_on_fatal: Optional[bool] = None,
        raise_argument_validation_error: Optional[bool] = None,
        final_answer_generator: Optional[FinalAnswerGenerator] = None,
    ) -> None:
        super().__init__(
            raise_argument_validation_error=bool(raise_argument_validation_error)
            if raise_argument_validation_error is not None
            else False
        )
        self.name = self.mcp_tool_name
        self.description = None
        self.arguments_schema = None
        self.settings = MCPToolSettings(
            endpoint=(
                endpoint if endpoint is not None
                else MCPToolSettings.endpoint
            ),
            auth_header=(
                auth_header if auth_header is not None
                else MCPToolSettings.auth_header
            ),
            transport=(
                transport if transport is not None
                else MCPToolSettings.transport
            ),
            max_concurrency=(
                max_concurrency if max_concurrency is not None
                else MCPToolSettings.max_concurrency
            ),
            response_char_limit=(
                response_char_limit if response_char_limit is not None
                else MCPToolSettings.response_char_limit
            ),
            enable_trace_logging=(
                enable_trace_logging if enable_trace_logging is not None
                else MCPToolSettings.enable_trace_logging
            ),
            raise_on_fatal=(
                raise_on_fatal if raise_on_fatal is not None
                else MCPToolSettings.raise_on_fatal
            ),
        )

        self._client: Any = None           # fastmcp.Client 实例
        self._connected = False            # 连接状态标记
        self._init_lock = asyncio.Lock()   # 防止 init() 并发重入
        self._conn_lock = asyncio.Lock()   # 保护底层连接操作
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        # Track configured semaphore capacity separately. Do not compare with
        # semaphore._value, because _value is runtime-available permits.
        self._semaphore_limit = self.settings.max_concurrency

    def dump_metadata(self) -> dict[str, Any]:
        base = super().dump_metadata()
        base.update(
            {
                "mcp_tool_name": self.mcp_tool_name,
                "endpoint": self.settings.endpoint,
                "transport": self.settings.transport,
                "max_concurrency": self.settings.max_concurrency,
                "response_char_limit": self.settings.response_char_limit,
                "enable_trace_logging": self.settings.enable_trace_logging,
                "raise_on_fatal": self.settings.raise_on_fatal,
                "auth_header_set": bool(self.settings.auth_header),
            }
        )
        return base
        self._final_answer_generator = final_answer_generator

    # ---- async context manager 支持 ----

    async def __aenter__(self) -> "MCPTool":
        """支持 `async with MCPTool(...) as tool:` 用法。"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出上下文时自动关闭连接。"""
        await self.close()

    # ---- 工厂方法 ----

    @classmethod
    def from_conf(cls, conf: Any | None = None, **overrides: Any) -> "MCPTool":
        """从 Hydra DictConfig / dict / 对象创建工具实例，overrides 优先级最高。"""
        conf = _resolve_conf(conf)
        params = {
            "endpoint": _read_conf(conf, "endpoint"),
            "auth_header": _read_conf(conf, "auth_header"),
            "transport": _read_conf(conf, "transport"),
            "max_concurrency": _read_conf(conf, "max_concurrency"),
            "response_char_limit": _read_conf(conf, "response_char_limit"),
            "enable_trace_logging": _read_conf(conf, "enable_trace_logging"),
            "raise_on_fatal": _read_conf(conf, "raise_on_fatal"),
            "raise_argument_validation_error": _read_conf(
                conf, "raise_argument_validation_error"
            ),
            "final_answer_generator": _read_conf(conf, "final_answer_generator"),
        }
        params.update(overrides)
        return cls(**params)

    # ---- 初始化 / 关闭 ----

    async def init(
        self,
        endpoint: Optional[str] = None,
        *,
        conf: Any | None = None,
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

            # 从 endpoint 参数更新
            if endpoint:
                self.settings.endpoint = endpoint

            # 从 conf 对象批量更新
            if conf is not None:
                conf = _resolve_conf(conf)
                self.settings.endpoint = _read_conf(
                    conf, "endpoint", self.settings.endpoint,
                )
                self.settings.auth_header = _read_conf(
                    conf, "auth_header", self.settings.auth_header,
                )
                self.settings.transport = _read_conf(
                    conf, "transport", self.settings.transport,
                )
                self.settings.max_concurrency = _coerce_int(
                    _read_conf(conf, "max_concurrency", self.settings.max_concurrency),
                    self.settings.max_concurrency,
                )
                self.settings.response_char_limit = _coerce_int(
                    _read_conf(conf, "response_char_limit", self.settings.response_char_limit),
                    self.settings.response_char_limit,
                    minimum=256,
                )
                self.settings.enable_trace_logging = bool(
                    _read_conf(conf, "enable_trace_logging", self.settings.enable_trace_logging)
                )
                self.settings.raise_on_fatal = bool(
                    _read_conf(conf, "raise_on_fatal", self.settings.raise_on_fatal)
                )

            # 显式 kwargs 优先级最高，覆盖一切
            if "auth_header" in kwargs:
                self.settings.auth_header = kwargs["auth_header"]
            if "transport" in kwargs:
                self.settings.transport = kwargs["transport"]
            if "max_concurrency" in kwargs:
                self.settings.max_concurrency = _coerce_int(
                    kwargs["max_concurrency"], self.settings.max_concurrency,
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

            # Rebuild semaphore only when configured capacity actually changes.
            # Avoid comparing with semaphore._value; that value fluctuates while
            # tasks are running and does not represent configured capacity.
            if self._semaphore_limit != self.settings.max_concurrency:
                self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)
                self._semaphore_limit = self.settings.max_concurrency

            await self._connect(trace_id=trace_id)
            await self._ensure_tool_exists(trace_id=trace_id)

    async def close(self) -> None:
        """优雅关闭 MCP 客户端连接。"""
        async with self._conn_lock:
            if self._client is None or not self._connected:
                return
            try:
                aexit = getattr(self._client, "__aexit__", None)
                if callable(aexit):
                    await aexit(None, None, None)
            except Exception as exc:
                # 【修复】原代码 close() 中没有异常日志，关闭失败时完全静默
                logger.warning("Error while closing MCP client: %s", exc)
            finally:
                self._connected = False
                self._client = None

    # ---- 公共调用接口 ----

    # async def run_batch(
    #     self,
    #     payloads: list[dict[str, Any]],
    #     *,
    #     trace_id: Optional[str] = None,
    #     fail_fast: bool = False,
    # ) -> list[str]:
    #     """批量并发调用 MCP 工具。"""
    #     ...

    # ---- 内部连接管理 ----

    async def _connect(self, *, trace_id: Optional[str] = None) -> None:
        """建立到 MCP 端点的连接（Streamable HTTP 或 SSE）。"""
        # 【修复】改为有意义的守卫：现在 fastmcp 用 try/except import，
        # Client/SSETransport 可能确实为 None
        if Client is None or SSETransport is None or StreamableHttpTransport is None:
            raise ToolFatalError(
                "fastmcp 未安装。请执行 `pip install 'fastmcp>=3.0.0'`。"
            )

        async with self._conn_lock:
            if self._connected and self._client is not None:
                return

            # 构造 HTTP headers（如有 auth token 则添加 Authorization）
            headers: dict[str, str] = {}
            if self.settings.auth_header:
                headers["Authorization"] = self.settings.auth_header

            transport = self._build_transport(headers=headers)
            self._client = Client(transport)

            # 尝试建立持久连接；若客户端支持 __aenter__ 则进入上下文
            try:
                aenter = getattr(self._client, "__aenter__", None)
                if callable(aenter):
                    await aenter()
                self._connected = True
                self._trace_log(
                    logging.INFO,
                    "Connected to MCP endpoint",
                    trace_id=trace_id,
                    endpoint=self.settings.endpoint,
                    transport=self.settings.transport,
                )
            except Exception as exc:
                self._connected = False
                self._client = None
                raise ToolFatalError(
                    f"Failed to connect MCP endpoint {self.settings.endpoint}: {exc}"
                ) from exc

    def _build_transport(self, *, headers: dict[str, str]) -> Any:
        transport_name = (self.settings.transport or "auto").strip().lower()
        if transport_name in {"", "auto"}:
            transport_name = "http"

        if transport_name in {"http", "streamable", "streamable_http", "streamable-http"}:
            transport_cls = StreamableHttpTransport
        elif transport_name in {"sse", "sse_transport"}:
            transport_cls = SSETransport
        else:
            raise ToolFatalError(
                f"Unsupported MCP transport '{self.settings.transport}'. "
                "Use 'http', 'sse', or 'auto'."
            )

        return transport_cls(
            url=self.settings.endpoint,
            headers=headers if headers else None,
        )

    async def _ensure_tool_exists(self, *, trace_id: Optional[str] = None) -> None:
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
            f"MCP tool '{self.mcp_tool_name}' not found at endpoint {self.settings.endpoint}."
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
            self.arguments_schema = self._coerce_schema(schema)

        self._trace_log(
            logging.DEBUG,
            "MCP tool metadata applied",
            trace_id=trace_id,
            description_set=self.description is not None,
            schema_set=self.arguments_schema is not None,
        )

    @staticmethod
    def _coerce_schema(value: Any) -> type[BaseModel] | None:
        """将 MCP schema 转成 pydantic BaseModel class（最小兼容）。"""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            value = parsed
        if isinstance(value, type) and issubclass(value, BaseModel):
            return value
        if isinstance(value, BaseModel):
            return value.__class__
        if isinstance(value, Mapping):
            schema = dict(value)
            model_name = schema.get("title") or "MCPArguments"

            class MCPArgumentsModel(BaseModel):
                model_config = ConfigDict(extra="allow")

                @classmethod
                def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
                    return schema

            MCPArgumentsModel.__name__ = str(model_name)
            return MCPArgumentsModel
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

        trace = trace_id or self._new_trace_id()

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
            except ToolError as exc:
                raise ToolRecoverableError from exc
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
        """如果响应超过 response_char_limit，进行截断或调用摘要生成器。"""
        if len(response_text) <= self.settings.response_char_limit:
            return response_text

        self._trace_log(
            logging.WARNING,
            "Response exceeded configured limit; generating condensed final answer",
            trace_id=trace_id,
            length=len(response_text),
            limit=self.settings.response_char_limit,
        )

        # 没有配置摘要生成器时，直接截断并提示
        if self._final_answer_generator is None:
            clipped = response_text[: self.settings.response_char_limit]
            return (
                f"{clipped}\n\n"
                "[Truncated] Response exceeded configured content limit. "
                "Configure `final_answer_generator` to auto-generate final answer."
            )

        # 调用摘要生成器（兼容同步/异步 callable）
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
            return self._recoverable_message(exc)

        self._trace_log(
            logging.ERROR, "Fatal MCP error",
            trace_id=trace_id, error=message,
        )
        fatal = ToolFatalError(message)
        if self.settings.raise_on_fatal:
            raise fatal from exc
        return self._recoverable_message(fatal)

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
    def _recoverable_message(exc: Exception) -> str:
        return f"[Tool] recoverable error: {exc}"

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
        if level == logging.DEBUG and not self.settings.enable_trace_logging:
            return
        payload = {"trace_id": trace_id or "-", **fields}
        logger.log(level, "%s | %s", message, payload)


# ---------------------------------------------------------------------------
# GenericMCPTool：开箱即用的 MCP 工具（无需子类化）
# ---------------------------------------------------------------------------

class GenericMCPTool(MCPTool):
    """通用 MCP 工具：直接将参数转发给指定的 `mcp_tool_name`。

    用法：
        tool = GenericMCPTool("search", endpoint="http://localhost:8100/mcp")
        await tool.init()
        result = await tool.run(query="hello world")
    """

    def __init__(self, mcp_tool_name: str, **kwargs: Any) -> None:
        # 先设置 mcp_tool_name，父类 __init__ 中可能间接使用
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
        """从配置创建 GenericMCPTool，mcp_tool_name 必须提供。"""
        conf = _resolve_conf(conf)
        resolved_name = (
            mcp_tool_name
            or _read_conf(conf, "mcp_tool_name")
            or _read_conf(conf, "name")
        )
        if not resolved_name:
            raise ValueError("`mcp_tool_name` is required for GenericMCPTool.from_conf().")
        params = {
            "endpoint": _read_conf(conf, "endpoint"),
            "auth_header": _read_conf(conf, "auth_header"),
            "transport": _read_conf(conf, "transport"),
            "max_concurrency": _read_conf(conf, "max_concurrency"),
            "response_char_limit": _read_conf(conf, "response_char_limit"),
            "enable_trace_logging": _read_conf(conf, "enable_trace_logging"),
            "raise_on_fatal": _read_conf(conf, "raise_on_fatal"),
            "raise_argument_validation_error": _read_conf(
                conf, "raise_argument_validation_error"
            ),
            "final_answer_generator": _read_conf(conf, "final_answer_generator"),
        }
        params.update(overrides)
        return cls(mcp_tool_name=resolved_name, **params)

    async def _run(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """子类实现：转发 MCP 调用。"""
        return await self._run_mcp_tool(arguments)


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
    tool = GenericMCPTool(
        mcp_tool_name=tool_name,
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
