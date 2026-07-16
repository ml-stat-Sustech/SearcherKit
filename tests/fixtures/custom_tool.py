from __future__ import annotations

from typing import Any

from searcherkit.tools import BaseTool, ToolConfig


class CustomTool(BaseTool):
    def __init__(self, *, config: ToolConfig, prefix: str, count: int) -> None:
        super().__init__(config=config)
        self.prefix = prefix
        self.count = count

    async def _run(self, *, query: str) -> tuple[str, dict[str, Any]]:
        return f"{self.prefix}:{query}", {"count": self.count}


NOT_A_TOOL = object()
