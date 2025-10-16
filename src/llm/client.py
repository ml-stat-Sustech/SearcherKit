from __future__ import annotations

import abc
import json
import os
from typing import Dict, List, Optional


ChatMessages = List[Dict[str, str]]


class LLMClient(abc.ABC):
    """Minimal chat-completion interface required by the WebWalker agent."""

    @abc.abstractmethod
    def complete(self, messages: ChatMessages, **kwargs) -> str:
        """Return the assistant reply for the provided conversation history."""


class EchoLLM(LLMClient):
    """
    Deterministic fallback model that mirrors the latest observation.

    Useful for offline testing when no real LLM credentials are available.
    """

    def complete(self, messages: ChatMessages, **kwargs) -> str:
        if not messages:
            return "Thought: No conversation context provided.\nFinal Answer: Unable to proceed."
        response_format = kwargs.get("response_format")
        last_message = messages[-1]["content"]
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            if "Accumulated Information" in last_message:
                return json.dumps({"judge": False})
            return json.dumps({"usefulness": False})
        return (
            "Thought: Using the observation to craft a draft answer.\n"
            f"Final Answer: {last_message}"
        )


class OpenAIChatClient(LLMClient):
    """Wrapper around the `openai` Python SDK."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("An OpenAI API key must be provided.")
        client_kwargs: Dict[str, object] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.default_kwargs = default_kwargs or {}

    def complete(self, messages: ChatMessages, **kwargs) -> str:
        payload = {**self.default_kwargs, **kwargs}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **payload,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


def build_llm_from_env() -> LLMClient:
    """
    Convenience factory that inspects environment variables.

    If `OPENAI_MODEL` is defined (or `OPENAI_MODEL_NAME`), an OpenAI-compatible client
    is returned. Otherwise a simple echo model is used as a safe fallback.
    """

    model = (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL_NAME")
        or os.environ.get("OPENAI_CHAT_MODEL")
    )
    if model:
        base_url = os.environ.get("OPENAI_MODEL_SERVER") or os.environ.get("OPENAI_BASE_URL")
        default_kwargs: Dict[str, object] = {}
        temperature = os.environ.get("OPENAI_TEMPERATURE")
        if temperature:
            default_kwargs["temperature"] = float(temperature)
        max_tokens = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
        if max_tokens:
            default_kwargs["max_tokens"] = int(max_tokens)
        return OpenAIChatClient(model=model, base_url=base_url, default_kwargs=default_kwargs)
    return EchoLLM()
