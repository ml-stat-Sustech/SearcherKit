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
    """Fetch documents from one of the configured data sources."""

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        source: DataSource | Mapping[str, DataSource],
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
        source: DataSource | Mapping[str, DataSource] | None = None,
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
            configured_source: DataSource | Mapping[str, DataSource]
            if len(config.source) == 1:
                configured_source = build_source(config.source[0])
            else:
                configured_source = {
                    source_name: build_source(source_name)
                    for source_name in config.source
                }
            self.__init__(
                configured_source,
                name=config.name,
                description=config.description,
                inputSchema=config.inputSchema or inputSchema,
                argument_mapping=config.argument_mapping or argument_mapping,
                response_char_limit=config.response_char_limit,
                summarizer=Summarizer(config=config.summarizer) if config.summarizer else None,
                summary_goal_key=config.summary_goal_key,
            )
            return
        if source is None:
            raise ValueError("VisitTool requires a source")
        self.source_map = dict(source) if isinstance(source, Mapping) else None
        if self.source_map is not None and not self.source_map:
            raise ValueError("VisitTool requires at least one source")
        super().__init__(
            name=name,
            description=description,
            inputSchema=inputSchema,
            argument_mapping=argument_mapping,
            summarizer=summarizer,
            summary_goal_key=summary_goal_key,
        )
        self.source = source if isinstance(source, DataSource) else None
        if self.source_map is not None:
            source_names = sorted(self.source_map)
            if description is None and self.description:
                choices = "\n".join(source_names)
                self.description = (
                    f"{self.description}\nChoose source to fetch from the list.\n{choices}"
                )
            if inputSchema is None:
                self.inputSchema = dict(self.inputSchema)
                self.inputSchema["required"] = [
                    *self.inputSchema.get("required", []),
                    "source",
                ]
        elif inputSchema is None:
            self.inputSchema = dict(self.inputSchema)
            self.inputSchema["properties"] = dict(self.inputSchema["properties"])
            self.inputSchema["properties"].pop("source", None)
        self.response_char_limit = (
            config.response_char_limit if config is not None else response_char_limit
        )
        if self.response_char_limit is not None and self.response_char_limit <= 0:
            raise ValueError(f"response_char_limit must be positive: {self.response_char_limit}")

    async def _run(
        self,
        *,
        document_id: str | list[str],
        goal: str | None = None,
        source: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Get the content of a document."""
        selected_source = self.source
        if self.source_map is not None:
            selected_source = self.source_map.get(source or "")
            if selected_source is None:
                available_sources = ", ".join(sorted(self.source_map))
                raise RecoverableError(
                    f"unknown source {source!r}; available sources: {available_sources}"
                )
        if selected_source is None:
            raise RuntimeError("VisitTool has no configured source")
        try:
            if isinstance(document_id, list):
                documents = list(
                    await asyncio.gather(
                        *[selected_source.fetch(did, goal=goal) for did in document_id]
                    )
                )
            else:
                documents = [await selected_source.fetch(document_id, goal=goal)]
        except SourceError as e:
            if self.source_map is not None:
                raise RecoverableError(str(e)) from e
            if "not found" in str(e).lower():
                return f"[Tool] {e}", {}
            raise RecoverableError(str(e)) from e
        except KeyError as e:
            return f"[Tool] Document not found: {document_id}", {}
        ret = []
        for document in documents:
            text = _format_document(document)
            if self.response_char_limit:
                return _limit_response(text, self.response_char_limit), _visit_extensions(
                    documents,
                    source=source if self.source_map is not None else None,
                )
            ret.append(text)
        return "\n=======\n".join(ret), _visit_extensions(
            documents,
            source=source if self.source_map is not None else None,
        )

    async def close(self) -> None:
        if self.source_map is not None:
            for source in self.source_map.values():
                await source.close()
        elif self.source is not None:
            await self.source.close()
