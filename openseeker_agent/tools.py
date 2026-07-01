"""MCP-backed OpenSeeker search/visit tool calls."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def _extract_text(response: Any) -> str:
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
    return "" if response is None else str(response)


async def call_mcp_async(tool_name: str, params: dict[str, Any]) -> str:
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport

    endpoint = os.environ.get("MCP_ENDPOINT")
    if not endpoint:
        raise RuntimeError("MCP_ENDPOINT is required for OpenSeeker BCP search/visit tools")

    transport_name = os.environ.get("MCP_TRANSPORT", "streamable-http")
    headers = None
    auth_header = os.environ.get("MCP_AUTH_HEADER")
    if auth_header:
        headers = {"Authorization": auth_header}

    if transport_name == "streamable-http":
        transport = StreamableHttpTransport(url=endpoint, headers=headers)
    elif transport_name == "sse":
        transport = SSETransport(url=endpoint, headers=headers)
    else:
        raise RuntimeError(f"unsupported MCP_TRANSPORT={transport_name!r}")

    async with Client(transport) as client:
        response = await client.call_tool(tool_name, params)
    return _extract_text(response)


def call_mcp(tool_name: str, params: dict[str, Any]) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call_mcp_async(tool_name, params))

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(call_mcp_async(tool_name, params))).result()


class Search:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg

    def call(self, params: dict[str, Any], **kwargs: Any) -> str:
        return call_mcp("search", params)


class Visit:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg

    def call(self, params: dict[str, Any], **kwargs: Any) -> str:
        return call_mcp("visit", params)

