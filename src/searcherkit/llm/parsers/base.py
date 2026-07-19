"""Parser interfaces and factory helpers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Protocol

from searcherkit.common.config import import_from_path
from searcherkit.common.messages import ChatMessage
from searcherkit.common.errors import LLMError


@dataclass
class ParserConfig:
    type: str = "upstream"
    target: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class ParsingError(LLMError):
    """Raised when model message payload cannot be parsed into `ChatMessage`."""


@dataclass(frozen=True, slots=True)
class LiveDeltaPart:
    """One parser-interpreted text fragment for Step-Level Live View."""

    field: Literal["content", "thinking", "final_answer"]
    text: str


class LiveDeltaSplitter(Protocol):
    """Parser-owned streaming text interpreter."""

    def feed(self, text: str) -> list[LiveDeltaPart]:
        """Interpret one raw provider content delta."""
        ...

    def flush(self) -> list[LiveDeltaPart]:
        """Return any buffered live fragments at stream end."""
        ...


class PlainLiveDeltaSplitter:
    """Default splitter for parsers whose streaming content has no template tags."""

    def feed(self, text: str) -> list[LiveDeltaPart]:
        if not text:
            return []
        return [LiveDeltaPart(field="content", text=text)]

    def flush(self) -> list[LiveDeltaPart]:
        return []


class Parser(abc.ABC):
    @property
    def uses_provider_tools(self) -> bool:
        return False

    def create_live_delta_splitter(self) -> LiveDeltaSplitter:
        """Create a parser-owned interpreter for raw streaming content deltas."""

        return PlainLiveDeltaSplitter()

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

    parser_type = config.get("type", "tongyi_deep_research") if isinstance(config, Mapping) else config.type
    if "upstream" in parser_type.lower():
        from searcherkit.llm.parsers.upstream import UpstreamParser

        return UpstreamParser()
    if "tongyi" in parser_type.lower() or "deepresearch" in parser_type.lower():
        from searcherkit.llm.parsers.tongyi_deep_research import TongyiDeepResearchParser

        return TongyiDeepResearchParser(config=config)
    if "webexplorer" in parser_type.lower():
        from searcherkit.llm.parsers.webexplorer import WebExplorerParser

        return WebExplorerParser()
    if "websailor" in parser_type.lower():
        from searcherkit.llm.parsers.websailor import WebSailorParser

        return WebSailorParser()

    raise ValueError(f"Cannot infer parser type from name: {parser_type}")
