from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload
import asyncio

from searcherkit.common.errors import RecoverableError
from searcherkit.sources import DataSource, Document, SourceError, build_source
from searcherkit.tools.base import BaseTool, ToolConfig
from searcherkit.tools.summarizer import Summarizer


def _visit_extensions(
    documents: list[Document],
    *,
    source: str | None = None,
) -> dict[str, object]:
    """Build tool extensions for visited documents."""
    items: list[dict[str, object]] = []
    for document in documents:
        item: dict[str, object] = {
            "id": document.id,
            "title": document.title,
            "url": document.url,
        }
        if source is not None:
            item["source"] = source
        items.append(item)
    return {"documents": items}


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
        argument_mapping: Mapping[str, str] | None = None,
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
        argument_mapping: Mapping[str, str] | None = None,
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
                inputSchema=config.inputSchema or inputSchema,
                argument_mapping=config.argument_mapping or argument_mapping,
                response_char_limit=config.response_char_limit,
                summarizer=Summarizer(config = config.summarizer) if config.summarizer else None,
                summary_goal_key=config.summary_goal_key
            )
            return
        if source is None:
            raise ValueError("VisitTool requires a source")
        super().__init__(
            name=name,
            description=description,
            inputSchema=inputSchema,
            argument_mapping=argument_mapping,
            summarizer=summarizer,
            summary_goal_key=summary_goal_key
        )
        self.source = source
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(
        self, *, document_id: str | list[str], goal: str | None = None
    ) -> tuple[str, dict[str, object]]:
        """Get the content of a document."""
        try:
            if isinstance(document_id, list):
                documents = list(
                    await asyncio.gather(
                        *[self.source.fetch(did, goal=goal) for did in document_id]
                    )
                )
            else:
                documents = [await self.source.fetch(document_id, goal=goal)]
        except SourceError as e:
            if "not found" in str(e).lower():
                return f"[Tool] {e}", {}
            raise RecoverableError(str(e)) from e
        except KeyError:
            return f"[Tool] Document not found: {document_id}", {}
        ret = []
        for document in documents:
            text = _format_document(document)
            if self.response_char_limit:
                return _limit_response(text, self.response_char_limit), _visit_extensions(
                    documents
                )
            ret.append(text)
        return "\n=======\n".join(ret), _visit_extensions(documents)

    async def close(self) -> None:
        await self.source.close()
