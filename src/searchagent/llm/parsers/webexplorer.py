"""WebExplorer prompt-format parser."""

from __future__ import annotations

from searchagent.llm.parsers.qwen import QwenParser


class WebExplorerParser(QwenParser):
    """Qwen-compatible parser configured for WebExplorer recipes."""

    def __init__(self) -> None:
        super().__init__(upstream_parsed=False, drop_thinking=False)
