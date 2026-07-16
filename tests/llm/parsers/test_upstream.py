from __future__ import annotations

import pytest

from searcherkit.common.messages import ToolCall, assistant, system, tool, user
from searcherkit.llm.parsers import ParserConfig, ParsingError, get_parser
from searcherkit.llm.parsers.upstream import UpstreamParser


def _assistant_payload(arguments: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "Need a lookup.",
        "reasoning_content": "Use the tool.",
        "tool_calls": [
            {
                "id": "call_lookup",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": arguments,
                },
            }
        ],
    }


@pytest.mark.parametrize("parser", [UpstreamParser(), get_parser(ParserConfig(type="upstream"))])
def test_upstream_parser_round_trips_provider_tool_messages(parser: UpstreamParser) -> None:
    model_messages = parser.to_model(
        [
            system("System."),
            user("Question?"),
            assistant(
                "Need a lookup.",
                thinking="Use the tool.",
                tool_calls=[ToolCall(id="call_lookup", name="lookup", arguments={"query": "next"})],
            ),
            tool({"call_lookup": "Result."}),
        ]
    )

    assert model_messages == [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Question?"},
        {
            "role": "assistant",
            "content": "Need a lookup.",
            "reasoning": "Use the tool.",
            "reasoning_content": "Use the tool.",
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query": "next"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_lookup", "content": "Result."},
    ]

    [parsed] = parser.from_model([_assistant_payload('{"query": "next"}')])

    assert parsed.role == "assistant"
    assert parsed.content == "Need a lookup."
    assert parsed.thinking == "Use the tool."
    assert parsed.tool_calls == [
        ToolCall(id="call_lookup", name="lookup", arguments={"query": "next"})
    ]


@pytest.mark.parametrize("arguments", ["not json", "[1, 2, 3]", "null"])
def test_upstream_parser_rejects_invalid_tool_call_arguments(arguments: str) -> None:
    parser = UpstreamParser()

    with pytest.raises(ParsingError):
        parser.from_model([_assistant_payload(arguments)])
