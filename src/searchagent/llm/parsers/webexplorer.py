"""WebExplorer prompt-format parser."""

from __future__ import annotations

from searchagent.llm.parsers.tongyi_deep_research import TongyiDeepResearchParser


class WebExplorerParser(TongyiDeepResearchParser):
    """Qwen-compatible parser configured for WebExplorer recipes."""

    def __init__(self) -> None:
        super().__init__()
