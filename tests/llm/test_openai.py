import asyncio
from collections.abc import Callable

import pytest
from openai import APITimeoutError

from searchagent.common.retry import RetryPolicy
from searchagent.llm.base import ClientConfig, OpenAIConfig
from searchagent.llm.openai import OpenAIClient
from mock_endpoints import upstream_parsed_endpoint


ClientFactory = Callable[[RetryPolicy | None], OpenAIClient]


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
            openai=OpenAIConfig(
                api_key="test-key",
                base_url=upstream_parsed_endpoint.BASE_URL,
                extra_client_kwargs={"max_retries": 0},
            ),
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
