from __future__ import annotations

from searcherkit.llm.parsers.base import PlainLiveDeltaSplitter
from searcherkit.llm.parsers.upstream import UpstreamParser


def _parts(splitter, chunks: list[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for chunk in chunks:
        result.extend((part.field, part.text) for part in splitter.feed(chunk))
    result.extend((part.field, part.text) for part in splitter.flush())
    return result


def test_upstream_live_splitter_enters_final_answer_across_chunks() -> None:
    assert _parts(
        UpstreamParser().create_live_delta_splitter(),
        ["before ", "<ans", "wer>", "Par", "is", "</answer>ignored"],
    ) == [
        ("content", "before "),
        ("final_answer", ""),
        ("final_answer", "Par"),
        ("final_answer", "is"),
    ]


def test_plain_live_splitter_does_not_interpret_answer_tags() -> None:
    assert _parts(PlainLiveDeltaSplitter(), ["<answer>Paris</answer>"]) == [
        ("content", "<answer>Paris</answer>"),
    ]
