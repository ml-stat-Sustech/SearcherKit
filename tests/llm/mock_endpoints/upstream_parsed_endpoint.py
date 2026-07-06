import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import respx

BASE_URL = "https://example.test/v1"
CONTENT = "Pong."
FINAL_CONTENT = "Final answer."
THINKING = "I should answer briefly."
TOOL_NAME = "lookup"
TOOL_ARGUMENTS = {"query": "next"}
TOOL_ARGUMENTS_JSON = json.dumps(TOOL_ARGUMENTS)
TOOL_CALL = {
    "id": "call_lookup",
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "arguments": TOOL_ARGUMENTS_JSON,
    },
}
USAGE = {
    "prompt_tokens": 9,
    "completion_tokens": 3,
    "total_tokens": 12,
}
HIGH_USAGE = {
    "prompt_tokens": 120,
    "completion_tokens": 10,
    "total_tokens": 130,
}
CONTEXT_LENGTH_ERROR = {
    "error": {
        "message": (
            "This model's maximum context length is 128000 tokens. However, "
            "your messages resulted in 130000 tokens. Please reduce the length "
            "of the messages."
        ),
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }
}


def _tool_call_response(*, usage: dict[str, int] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": CONTENT,
                        "reasoning_content": THINKING,
                        "tool_calls": [TOOL_CALL],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": usage or USAGE,
        },
    )


def _final_response(*, usage: dict[str, int] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test-final",
            "object": "chat.completion",
            "created": 1_700_000_001,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": FINAL_CONTENT,
                        "reasoning_content": THINKING,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage or USAGE,
        },
    )


def _context_length_response() -> httpx.Response:
    return httpx.Response(400, json=CONTEXT_LENGTH_ERROR)


@contextmanager
def context(
    captured_payload: dict[str, Any] | None = None,
    *,
    assert_all_called: bool = True,
    outcomes: list[str] | tuple[str, ...] = ("success",),
) -> Iterator[respx.Router]:
    remaining = list(outcomes)

    def handler(request: httpx.Request) -> httpx.Response:
        if captured_payload is not None:
            captured_payload.clear()
            captured_payload.update(json.loads(request.content))
        outcome = remaining.pop(0) if remaining else outcomes[-1]
        if outcome in {"success", "tool_call"}:
            return _tool_call_response()
        if outcome == "tool_call_high_usage":
            return _tool_call_response(usage=HIGH_USAGE)
        if outcome == "final":
            return _final_response()
        if outcome == "final_high_usage":
            return _final_response(usage=HIGH_USAGE)
        if outcome == "context_length":
            return _context_length_response()
        if outcome == "timeout":
            raise httpx.TimeoutException("mock upstream timeout", request=request)
        raise ValueError(f"Unknown mock outcome: {outcome}")

    with respx.mock(assert_all_called=assert_all_called) as router:
        router.post(f"{BASE_URL}/chat/completions").mock(side_effect=handler)
        yield router
