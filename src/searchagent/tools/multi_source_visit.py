"""Native multi-source visit tool backed by configured data sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload

from searchagent.common.errors import RecoverableError
from searchagent.sources import DataSource, Document, SourceError, build_source
from searchagent.tools.base import BaseTool, ToolConfig

VISIT_DESCRIPTION = """
Fetch a document from one of the configured data sources.
Choose source to fetch from the list.
{sources}
"""

VISIT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_id": {"type": "string"},
        "goal": {"type": "string"},
        "source": {"type": "string"},
    },
    "required": ["document_id", "source"],
    "additionalProperties": False,
}


def _format_document(document: Document) -> str:
    text = ""
    text += f"[{document.title}]({document.url})\n"
    text += f"{document.text}\n"
    if document.metadata:
        text += f"Metadata: {document.metadata}\n"

    return text


def _limit_response(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Truncated] Response exceeded configured content limit."


class MultiSourceVisitTool(BaseTool):
    """Fetch a document from one of the configured data sources."""

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        source_map: dict[str, DataSource],
        *,
        name: str = "visit",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        source_map: dict[str, DataSource] | None = None,
        *,
        name: str = "visit",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
        config: ToolConfig | None = None,
    ) -> None:
        if config:
            if not getattr(config, "source", None):
                raise ValueError("MultiSourceVisitTool requires a list of sources to be created from a tool config")
            source_names = getattr(config, "source")
            if not isinstance(source_names, list):
                raise ValueError("MultiSourceVisitTool requires a list of sources")
            self.__init__(
                {s: build_source(s) for s in source_names},
                name=config.name,
                description=config.description,
                inputSchema=config.inputSchema or inputSchema,
                response_char_limit=config.response_char_limit,
            )
            return

        if not source_map:
            raise ValueError("MultiSourceVisitTool requires a dict of sources")
        source_names = sorted(source_map.keys())
        rendered_description = description or VISIT_DESCRIPTION.format(
            sources="\n".join(name for name in source_names)
        )
        super().__init__(
            name=name,
            description=rendered_description,
            inputSchema=inputSchema or VISIT_INPUT_SCHEMA,
        )
        self.source_map = source_map
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, **kwargs: Any) -> str:
        document_id = str(kwargs["document_id"])
        raw_goal = kwargs.get("goal")
        goal = str(raw_goal) if raw_goal is not None else None
        source_name = str(kwargs["source"])
        source = self.source_map.get(source_name)
        if source is None:
            available_sources = ", ".join(sorted(self.source_map.keys()))
            raise RecoverableError(
                f"unknown source {source_name!r}; available sources: {available_sources}"
            )
        try:
            document = await source.fetch(document_id, goal=goal)
        except SourceError as e:
            raise RecoverableError(str(e)) from e
        except KeyError as e:
            raise RecoverableError(f"Document not found: {document_id}") from e
        text = _format_document(document)
        if self.response_char_limit is not None:
            text = _limit_response(text, self.response_char_limit)
        return text

    async def close(self) -> None:
        for source in self.source_map.values():
            await source.close()
