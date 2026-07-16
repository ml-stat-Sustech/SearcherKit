"""MediaWiki dump reading and preprocessing for the local wiki plugin."""

from __future__ import annotations

import bz2
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from searcherkit.plugins.indexing import IndexDocument


WIKI_NAMESPACE = "{http://www.mediawiki.org/xml/export-0.11/}"
SECTION_STOP_MARKERS = (
    "\n## See also",
    "\n## References",
    "\n## Further reading",
    "\n## External links",
    "\n[Category:",
)


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _plain_text_from_wikitext(wikitext: str) -> tuple[list[Mapping[str, str]], str]:
    links: list[Mapping[str, str]] = []

    def replace_link(match: re.Match[str]) -> str:
        raw = match.group(1)
        if ":" in raw and raw.split(":", 1)[0] in {"File", "Image", "Category"}:
            return ""
        target, _, label = raw.partition("|")
        text = label or target
        text = text.strip()
        target = target.strip()
        if text and target:
            links.append({"text": text, "target": target})
        return text

    text = re.sub(r"\[\[([^\]]+)\]\]", replace_link, wikitext)
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref\b[^>/]*/>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref\b.*?</ref>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"## \1", text, flags=re.MULTILINE)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)

    split_index = -1
    for marker in SECTION_STOP_MARKERS:
        marker_index = text.find(marker)
        if marker_index != -1 and (split_index == -1 or marker_index < split_index):
            split_index = marker_index
    if split_index != -1:
        text = text[:split_index]

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return links, text.strip()


def preprocess_wiki_page(
    *,
    title: str,
    wikitext: str,
    base_url: str = "https://en.wikipedia.org/wiki/",
) -> IndexDocument | None:
    title = title.strip()
    if not title or title.startswith("Wikipedia:"):
        return None
    links, text = _plain_text_from_wikitext(wikitext)
    if not text:
        return None
    slug = quote(title.replace(" ", "_"))
    return IndexDocument(
        id=slug,
        title=title,
        text=text,
        url=f"{base_url.rstrip('/')}/{slug}",
        links=links,
        metadata={"source": "wiki"},
    )


class WikiDumpSource:
    """Iterate normalized documents from a MediaWiki XML dump."""

    def __init__(self, dump_path: str | Path, *, base_url: str = "https://en.wikipedia.org/wiki/") -> None:
        self.dump_path = Path(dump_path)
        self.base_url = base_url

    def iter_documents(self, *, limit: int | None = None) -> Iterator[IndexDocument]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        if not self.dump_path.exists():
            raise FileNotFoundError(self.dump_path)

        count = 0
        with self._open_dump() as handle:
            context = ET.iterparse(handle, events=("end",))
            for _, elem in context:
                if _strip_namespace(elem.tag) != "page":
                    continue
                document = self._document_from_page(elem)
                elem.clear()
                if document is None:
                    continue
                yield document
                count += 1
                if limit is not None and count >= limit:
                    return

    def _open_dump(self) -> Any:
        if self.dump_path.suffix == ".bz2":
            return bz2.open(self.dump_path, "rb")
        return self.dump_path.open("rb")

    def _document_from_page(self, page: ET.Element) -> IndexDocument | None:
        title_elem = page.find(f"./{WIKI_NAMESPACE}title")
        if title_elem is None:
            title_elem = page.find("./title")
        ns_elem = page.find(f"./{WIKI_NAMESPACE}ns")
        if ns_elem is None:
            ns_elem = page.find("./ns")
        redirect_elem = page.find(f"./{WIKI_NAMESPACE}redirect")
        if redirect_elem is None:
            redirect_elem = page.find("./redirect")
        text_elem = page.find(f"./{WIKI_NAMESPACE}revision/{WIKI_NAMESPACE}text")
        if text_elem is None:
            text_elem = page.find("./revision/text")

        if ns_elem is None or ns_elem.text != "0" or redirect_elem is not None:
            return None
        if title_elem is None or not title_elem.text:
            return None
        if text_elem is None or not text_elem.text:
            return None
        return preprocess_wiki_page(
            title=title_elem.text,
            wikitext=text_elem.text,
            base_url=self.base_url,
        )
