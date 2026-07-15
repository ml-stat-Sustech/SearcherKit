import asyncio
from collections.abc import Callable

import pytest
from anthropic import APITimeoutError

from searchagent.common.retry import RetryPolicy
from searchagent.llm.base import ClientConfig
from searchagent.llm.anthropic_client import AnthropicClient
from mock_endpoints import anthropic


ClientFactory = Callable[[RetryPolicy | None], AnthropicClient]


def _retry_policy(*exceptions: type[Exception]) -> RetryPolicy:
    return RetryPolicy(
        max_tries=2,
        exceptions=exceptions,
        jitter=None,
        factor=0.0,
    )


def _direct_client(retry_policy: RetryPolicy | None = None) -> AnthropicClient:
    return AnthropicClient(
        model="claude-test",
        api_key="test-key",
        base_url=anthropic.BASE_URL,
        default_kwargs={"max_tokens": 128},
        retry_policy=retry_policy,
        max_retries=0,
    )


def _config_client(retry_policy: RetryPolicy | None = None) -> AnthropicClient:
    return AnthropicClient(
        config=ClientConfig(
            type="anthropic",
            model="claude-test",
            retry_policy=retry_policy,
            default_kwargs={"max_tokens": 128},
            api_key="test-key",
            base_url=anthropic.BASE_URL,
            extra_client_kwargs={"max_retries": 0},
        )
    )


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_complete_fn(client_factory: ClientFactory) -> None:
    async def run() -> None:
        with anthropic.context():
            client = client_factory(None)

            message = await client.complete([{"role": "user", "content": "Go"}])

        assert message["role"] == "assistant"
        assert message["content"] == anthropic.CONTENT
        assert message["tool_calls"] == [anthropic.MESSAGE_TOOL_CALL]
        assert message["stop_reason"] == "tool_use"

    asyncio.run(run())


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_complete_with_usage_fn(client_factory: ClientFactory) -> None:
    async def run() -> None:
        captured_payload: dict[str, object] = {}

        with anthropic.context(captured_payload):
            client = client_factory(None)

            message, usage = await client.complete_with_usage(
                [{"role": "user", "content": "Ping?"}]
            )

        assert captured_payload == {
            "model": "claude-test",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Ping?"}],
                }
            ],
            "max_tokens": 128,
        }
        assert message["role"] == "assistant"
        assert message["content"] == anthropic.CONTENT
        assert message["tool_calls"] == [anthropic.MESSAGE_TOOL_CALL]
        assert message["stop_reason"] == "tool_use"
        assert usage == anthropic.USAGE

    asyncio.run(run())


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_retry_success(client_factory: ClientFactory) -> None:
    async def run() -> None:
        with anthropic.context(outcomes=("timeout", "success")) as router:
            client = client_factory(_retry_policy(APITimeoutError))

            message, usage = await client.complete_with_usage(
                [{"role": "user", "content": "Ping?"}]
            )

            assert len(router.calls) == 2
            assert message["content"] == anthropic.CONTENT
            assert usage == anthropic.USAGE

    asyncio.run(run())


@pytest.mark.parametrize("client_factory", [_direct_client, _config_client])
def test_retry_failure(client_factory: ClientFactory) -> None:
    async def run() -> None:
        with anthropic.context(
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
