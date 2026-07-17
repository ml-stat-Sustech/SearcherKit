from __future__ import annotations

from searcherkit.common.messages import tool
from searcherkit.llm.parsers import Parser, UpstreamParser
from searcherkit.llm.parsers import TongyiDeepResearchParser


def _parts(parser: Parser, chunks: list[str]) -> list[tuple[str, str]]:
    splitter = parser.create_live_delta_splitter()
    result: list[tuple[str, str]] = []
    for chunk in chunks:
        result.extend((part.field, part.text) for part in splitter.feed(chunk))
    result.extend((part.field, part.text) for part in splitter.flush())
    return result


def test_plain_parser_live_delta_splitter_emits_content() -> None:
    class PlainParser(Parser):
        def from_model(self, messages):
            return []

        def to_model(self, messages):
            return []

    assert _parts(PlainParser(), ["hel", "lo"]) == [("content", "hel"), ("content", "lo")]


def test_qwen_live_delta_splitter_interprets_thinking_and_tool_calls_across_chunks() -> None:
    parts = _parts(
        TongyiDeepResearchParser(),
        [
            "pre <thi",
            "nk>reason",
            "ing</think><tool_call>",
            '{"name":"search"}',
            "</tool_call><answer>Paris</answer>",
        ],
    )

    assert parts == [
        ("content", "pre "),
        ("thinking", "reason"),
        ("thinking", "ing"),
        ("content", "<answer>Paris</answer>"),
    ]


def test_upstream_parser_streaming_uses_plain_content_splitter() -> None:
    parts = _parts(UpstreamParser(), ["<think>raw</think>"])

    assert parts == [("content", "<think>raw</think>")]


def test_qwen_live_delta_splitter_handles_closing_tag_across_chunks() -> None:
    parts = _parts(TongyiDeepResearchParser(), ["<think>reason</thi", "nk>answer"])

    assert parts == [("thinking", "reason"), ("content", "answer")]


def test_qwen_live_delta_splitter_flushes_incomplete_tag_as_content() -> None:
    parts = _parts(TongyiDeepResearchParser(), ["before <thi"])

    assert parts == [("content", "before "), ("content", "<thi")]


def test_qwen_parser_preserves_answer_tag_for_toolless_assistant_message() -> None:
    message = next(
        iter(
            TongyiDeepResearchParser().from_model(
                [
                    {
                        "role": "assistant",
                        "content": "reasoning summary <answer>Paris</answer> trailing note",
                    }
                ]
            )
        )
    )

    assert message.content == "reasoning summary <answer>Paris</answer> trailing note"
    assert message.tool_calls == []


def test_qwen_parser_keeps_answer_tags_when_tool_calls_are_present() -> None:
    message = next(
        iter(
            TongyiDeepResearchParser().from_model(
                [
                    {
                        "role": "assistant",
                        "content": (
                            'Use this later: <answer>Paris</answer><tool_call>{"name":"search",'
                            '"arguments":{"query":"demo"}}</tool_call>'
                        ),
                    }
                ]
            )
        )
    )

    assert message.content == "Use this later: <answer>Paris</answer>"
    assert message.tool_calls is not None
    assert message.tool_calls[0].name == "search"


def test_qwen_parser_renders_upstream_tool_response_mapping() -> None:
    [message] = TongyiDeepResearchParser().to_model([tool({"call-a": "first", "call-b": "second"})])

    assert message == {
        "role": "user",
        "content": (
            "<tool_response>first</tool_response>\n"
            "<tool_response>second</tool_response>"
        ),
    }


def test_upstream_parser_preserves_answer_tag_for_toolless_assistant_message() -> None:
    message = next(
        iter(
            UpstreamParser().from_model(
                [{"role": "assistant", "content": "<answer>Paris</answer>"}]
            )
        )
    )

    assert message.content == "<answer>Paris</answer>"
