"""Parser interfaces and factory helpers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from searchagent.common.config import import_from_path
from searchagent.common.messages import ChatMessage
from searchagent.errors import LLMError


@dataclass
class QwenParserConfig:
    upstream_parsed: bool = field(default=False)
    drop_thinking: bool = field(default=True)


@dataclass
class ParserConfig:
    type: str = "qwen"
    target: str | None = None
    qwen: QwenParserConfig | None = field(default=None)
    kwargs: dict[str, Any] = field(default_factory=dict)


class ParsingError(LLMError):
    """Raised when model message payload cannot be parsed into `ChatMessage`."""


class Parser(abc.ABC):
    @property
    def uses_provider_tools(self) -> bool:
        return False

    @abc.abstractmethod
    def from_model(self, messages: Iterable[dict[str, Any]]) -> Iterable[ChatMessage]:
        """Parse provider/model messages into internal chat messages."""

    @abc.abstractmethod
    def to_model(self, messages: Iterable[ChatMessage]) -> Iterable[dict[str, Any]]:
        """Render internal chat messages into provider/model messages."""


def get_parser(config: ParserConfig | Mapping[str, Any]) -> Parser:
    target = config.get("target") if isinstance(config, Mapping) else config.target
    if target:
        parser_cls = import_from_path(target)
        kwargs = config.get("kwargs", {}) if isinstance(config, Mapping) else config.kwargs
        parser = parser_cls(**(kwargs or {}))
        if not isinstance(parser, Parser):
            raise TypeError(f"Parser target must construct a Parser, got {type(parser)}")
        return parser

    parser_type = config.get("type", "qwen") if isinstance(config, Mapping) else config.type
    if "qwen" in parser_type.lower():
        from searchagent.llm.parsers.qwen import QwenParser

        return QwenParser(config=config)
    if "webexplorer" in parser_type.lower():
        from searchagent.llm.parsers.webexplorer import WebExplorerParser

        return WebExplorerParser()
    if "websailor" in parser_type.lower():
        from searchagent.llm.parsers.websailor import WebSailorParser

        return WebSailorParser()

    raise ValueError(f"Cannot infer parser type from name: {parser_type}")
