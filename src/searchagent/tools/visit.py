from __future__ import annotations

from typing import Any, Mapping, overload
import json

from searchagent.common.errors import RecoverableError
from searchagent.sources import DataSource, Document, SourceError, build_source
from searchagent.tools.base import BaseTool, SummarizerConfig, ToolConfig
from searchagent.tools.summarizer import Summarizer

VISIT_DESCRIPTION = """
Get the content of a document.
"""

VISIT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_id": {"type": "string"},
        "goal": {"type": "string"},
    },
    "required": ["document_id"],
    "additionalProperties": False,
}


def _limit_response(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Truncated] Response exceeded configured content limit."

def _format_document(document: Document) -> str:
    text = ""
    text += f"[{document.title}]({document.url})\n"
    text += f"{document.text}\n"
    # if document.metadata:
    #     text += f"Metadata: {json.dumps(document.metadata, ensure_ascii=False, indent=None)}\n"

    return text

class VisitTool(BaseTool):
    """Fetch one document from a configured data source."""

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        source: DataSource,
        *,
        name: str = "visit",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
        summarizer: Summarizer | None = None,
        summary_goal_key = "goal",
    ) -> None: ...

    def __init__(
        self,
        source: DataSource | None = None,
        *,
        name: str = "visit",
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        response_char_limit: int | None = None,
        summarizer: Summarizer | None = None,
        summary_goal_key = "goal",
        config: ToolConfig | None = None,
    ) -> None:
        if config:
            if not config.source:
                raise ValueError("VisitTool requires a source to be created from a tool config")
            if len(config.source) != 1:
                raise ValueError("VisitTool requires exactly one source")
            self.__init__(
                build_source(config.source[0]),
                name = config.name,
                description=config.description,
                inputSchema=inputSchema,
                response_char_limit=config.response_char_limit,
                summarizer=Summarizer(config = config.summarizer) if config.summarizer else None,
                summary_goal_key=summary_goal_key
            )
            return
        if source is None:
            raise ValueError("VisitTool requires a source")
        super().__init__(
            name=name,
            description=description or VISIT_DESCRIPTION,
            inputSchema=inputSchema or VISIT_INPUT_SCHEMA,
            summarizer=summarizer,
            summary_goal_key=summary_goal_key
        )
        self.source = source
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(self, **kwargs: Any) -> str:
        document_id = str(kwargs["document_id"])
        raw_goal = kwargs.get("goal")
        goal = str(raw_goal) if raw_goal is not None else None
        try:
            document = await self.source.fetch(document_id, goal=goal)
        except SourceError as e:
            raise RecoverableError(str(e)) from e
        except KeyError as e:
            raise RecoverableError(f"Document not found: {document_id}") from e
        text = _format_document(document)
        if self.response_char_limit:
            return _limit_response(text, self.response_char_limit)
        return text

    async def close(self) -> None:
        await self.source.close()
