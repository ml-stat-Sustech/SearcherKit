import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
import respx
from openai import APITimeoutError

from searchagent.common.retry import RetryPolicy
from searchagent.llm.base import Client, ClientConfig, LLMStreamChunk
from searchagent.llm.openai_client import OpenAIClient
from mock_endpoints import upstream_parsed_endpoint


ClientFactory = Callable[[RetryPolicy | None], OpenAIClient]


def _stream_response() -> httpx.Response:
    events = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Hel",
                        "reasoning_content": "th",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_search",
                                "type": "function",
                                "function": {"name": "sea", "arguments": '{"q":"'},
                            },
                            {
                                "index": 1,
                                "id": "call_visit",
                                "type": "function",
                                "function": {"name": "vis", "arguments": '{"id":'},
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "lo",
                        "reasoning_content": "ink",
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {"name": "it", "arguments": "1}"},
                            },
                            {
                                "index": 0,
                                "function": {"name": "rch", "arguments": 'x"}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "gpt-test",
            "choices": [],
            "usage": upstream_parsed_endpoint.USAGE,
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    return httpx.Response(
        200,
        content=content.encode(),
        headers={"content-type": "text/event-stream"},
    )


def _retry_policy(*exceptions: type[Exception]) -> RetryPolicy:
    return RetryPolicy(
        max_tries=2,
        exceptions=exceptions,
        jitter=None,
        factor=0.0,
    )


def _direct_client(retry_policy: RetryPolicy | None = None) -> OpenAIClient:
    return OpenAIClient(
        model="gpt-test",
        api_key="test-key",
        base_url=upstream_parsed_endpoint.BASE_URL,
        retry_policy=retry_policy,
        default_kwargs={"temperature": 0.7, "max_tokens": 64},
        max_retries=0,
    )


def _config_client(retry_policy: RetryPolicy | None = None) -> OpenAIClient:
    return OpenAIClient(
        config=ClientConfig(
            type="openai",
            model="gpt-test",
            retry_policy=retry_policy,
            default_kwargs={"temperature": 0.7, "max_tokens": 64},
            api_key="test-key",
            base_url=upstream_parsed_endpoint.BASE_URL,
            extra_client_kwargs={"max_retries": 0},
        )
    )


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_complete_fn(client_factory: ClientFactory) -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context():
            client = client_factory(None)

            message = await client.complete([{"role": "user", "content": "Go"}])

        assert message["role"] == "assistant"
        assert message["content"] == upstream_parsed_endpoint.CONTENT
        assert message["reasoning_content"] == upstream_parsed_endpoint.THINKING
        assert message["refusal"] is None
        assert message["tool_calls"] == [upstream_parsed_endpoint.TOOL_CALL]

    asyncio.run(run())


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_complete_with_usage_fn(client_factory: ClientFactory) -> None:
    async def run() -> None:
        captured_payload: dict[str, object] = {}

        with upstream_parsed_endpoint.context(captured_payload):
            client = client_factory(None)

            message, usage = await client.complete_with_usage(
                [{"role": "user", "content": "Ping?"}],
                temperature=0.2,
            )

        assert captured_payload == {
            "messages": [{"role": "user", "content": "Ping?"}],
            "model": "gpt-test",
            "temperature": 0.2,
            "max_tokens": 64,
        }
        assert message["role"] == "assistant"
        assert message["content"] == upstream_parsed_endpoint.CONTENT
        assert message["reasoning_content"] == upstream_parsed_endpoint.THINKING
        assert message["refusal"] is None
        assert message["tool_calls"] == [upstream_parsed_endpoint.TOOL_CALL]
        assert usage is not None
        assert usage.prompt_tokens == upstream_parsed_endpoint.USAGE["prompt_tokens"]
        assert usage.completion_tokens == upstream_parsed_endpoint.USAGE["completion_tokens"]
        assert usage.total_tokens == upstream_parsed_endpoint.USAGE["total_tokens"]

    asyncio.run(run())


def test_stream_complete_with_usage_assembles_native_sse_chunks() -> None:
    async def run() -> None:
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content))
            return _stream_response()

        client = _direct_client(None)
        with respx.mock(assert_all_called=True) as router:
            router.post(f"{upstream_parsed_endpoint.BASE_URL}/chat/completions").mock(
                side_effect=handler
            )
            chunks = [
                chunk
                async for chunk in client.stream_complete_with_usage(
                    [{"role": "user", "content": "Stream?"}],
                    temperature=0.2,
                    stream=False,
                )
            ]

        assert captured_payload == {
            "messages": [{"role": "user", "content": "Stream?"}],
            "model": "gpt-test",
            "temperature": 0.2,
            "max_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert client.default_kwargs == {"temperature": 0.7, "max_tokens": 64}
        assert [
            (chunk.content_delta, chunk.thinking_delta, chunk.done)
            for chunk in chunks
        ] == [
            ("Hel", "th", False),
            ("lo", "ink", False),
            ("", "", True),
        ]

        terminal = chunks[-1]
        assert terminal.message == {
            "role": "assistant",
            "content": "Hello",
            "reasoning": "think",
            "reasoning_content": "think",
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"x"}'},
                },
                {
                    "id": "call_visit",
                    "type": "function",
                    "function": {"name": "visit", "arguments": '{"id":1}'},
                },
            ],
        }
        assert terminal.usage is not None
        assert terminal.usage.total_tokens == upstream_parsed_endpoint.USAGE["total_tokens"]

    asyncio.run(run())


def test_stream_complete_with_usage_preserves_explicit_stream_options() -> None:
    async def run() -> None:
        default_options = {"custom": "default"}
        explicit_options = {"include_usage": False, "custom": "explicit"}
        captured_payloads: list[dict[str, object]] = []

        client = OpenAIClient(
            model="gpt-test",
            api_key="test-key",
            base_url=upstream_parsed_endpoint.BASE_URL,
            default_kwargs={"stream_options": default_options},
        )

        async def empty_stream():
            if False:
                yield None

        async def create_completion(messages, payload, session_id=None):
            captured_payloads.append(payload)
            return empty_stream()

        client._create_completion = create_completion

        first = [
            chunk
            async for chunk in client.stream_complete_with_usage(
                [{"role": "user", "content": "Default options?"}]
            )
        ]
        second = [
            chunk
            async for chunk in client.stream_complete_with_usage(
                [{"role": "user", "content": "Explicit options?"}],
                stream_options=explicit_options,
            )
        ]

        assert first[-1].done is True
        assert second[-1].done is True
        assert captured_payloads[0]["stream_options"] == {
            "custom": "default",
            "include_usage": True,
        }
        assert captured_payloads[1]["stream_options"] == explicit_options
        assert captured_payloads[0]["stream_options"] is not default_options
        assert captured_payloads[1]["stream_options"] is not explicit_options
        assert default_options == {"custom": "default"}
        assert explicit_options == {"include_usage": False, "custom": "explicit"}
        assert client.default_kwargs == {"stream_options": default_options}

    asyncio.run(run())


def test_client_stream_fallback_yields_one_terminal_chunk() -> None:
    class FallbackClient(Client):
        async def complete(self, messages, session_id=None, **kwargs):
            return (await self.complete_with_usage(messages, session_id, **kwargs))[0]

        async def complete_with_usage(self, messages, session_id=None, **kwargs):
            return {"role": "assistant", "content": "fallback"}, {"total_tokens": 3}

    async def run() -> None:
        chunks = [
            chunk
            async for chunk in FallbackClient().stream_complete_with_usage(
                [{"role": "user", "content": "Fallback?"}]
            )
        ]

        assert chunks == [
            LLMStreamChunk(
                message={"role": "assistant", "content": "fallback"},
                usage={"total_tokens": 3},
                done=True,
            )
        ]

    asyncio.run(run())


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_retry_success(client_factory: ClientFactory) -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context(outcomes=("timeout", "success")) as router:
            client = client_factory(_retry_policy(APITimeoutError))

            message, usage = await client.complete_with_usage(
                [{"role": "user", "content": "Ping?"}]
            )

            assert len(router.calls) == 2
            assert message["content"] == upstream_parsed_endpoint.CONTENT
            assert usage is not None
            assert usage.total_tokens == upstream_parsed_endpoint.USAGE["total_tokens"]

    asyncio.run(run())


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_retry_failure(client_factory: ClientFactory) -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context(
            outcomes=("timeout", "timeout"),
            assert_all_called=False,
        ) as router:
            client = client_factory(_retry_policy(APITimeoutError))

            with pytest.raises(APITimeoutError):
                await client.complete_with_usage(
                    [{"role": "user", "content": "Ping?"}]
                )

            assert len(router.calls) == 2

    asyncio.run(run())
