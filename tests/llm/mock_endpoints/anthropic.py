import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import respx

BASE_URL = "https://example.test"
CONTENT = "Pong."
THINKING = "I should answer briefly."
TOOL_NAME = "lookup"
TOOL_ARGUMENTS = {"query": "next"}
TOOL_CALL = {
    "id": "toolu_lookup",
    "type": "tool_use",
    "name": TOOL_NAME,
    "input": TOOL_ARGUMENTS,
}
MESSAGE_TOOL_CALL = {
    "id": "toolu_lookup",
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "arguments": json.dumps(TOOL_ARGUMENTS),
    },
}
USAGE = {
    "prompt_tokens": 13,
    "completion_tokens": 4,
    "total_tokens": 17,
}


def _success_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [
                {
                    "type": "thinking",
                    "thinking": THINKING,
                    "signature": "test-signature",
                },
                {"type": "text", "text": CONTENT},
                TOOL_CALL,
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 13, "output_tokens": 4},
        },
    )


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
        if outcome == "success":
            return _success_response()
        if outcome == "timeout":
            raise httpx.TimeoutException("mock anthropic timeout", request=request)
        raise ValueError(f"Unknown mock outcome: {outcome}")

    with respx.mock(assert_all_called=assert_all_called) as router:
        router.post(f"{BASE_URL}/v1/messages").mock(side_effect=handler)
        yield router
