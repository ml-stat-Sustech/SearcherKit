from __future__ import annotations

import re

from searchagent.common.errors import SearchAgentError


def _openai_provider_error_types() -> tuple[type[BaseException], ...]:
    try:
        import openai
    except ImportError:
        return ()
    names = (
        "APIError",
        "APIConnectionError",
        "APIResponseValidationError",
        "APIStatusError",
        "APITimeoutError",
        "AuthenticationError",
        "BadRequestError",
        "ConflictError",
        "InternalServerError",
        "NotFoundError",
        "PermissionDeniedError",
        "RateLimitError",
        "UnprocessableEntityError",
    )
    types: list[type[BaseException]] = []
    for name in names:
        error_type = getattr(openai, name, None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            types.append(error_type)
    return tuple(types)


def _provider_error_message(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        message = body.get("message")
        if isinstance(message, str):
            return message
    return str(exc)


def _missing_model_from_provider_message(message: str) -> str | None:
    match = re.search(r"model ['\"]([^'\"]+)['\"] not found", message, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _looks_like_non_chat_model_error(message: str) -> bool:
    lowered = message.lower()
    if "chat" not in lowered and "completion" not in lowered:
        return False
    return any(marker in lowered for marker in ("does not support", "not support", "embedding", "embed"))


def friendly_provider_error_message(exc: BaseException, *, model_label: str = "provider/model") -> str:
    status_code = getattr(exc, "status_code", None)
    provider_message = _provider_error_message(exc)
    missing_model = _missing_model_from_provider_message(provider_message) or model_label
    if status_code == 404:
        return (
            f"Model not found: {missing_model}. Use /models to choose an available model, "
            "or install/start this model in the provider before retrying."
        )
    if status_code == 400 and _looks_like_non_chat_model_error(provider_message):
        return f"Model cannot be used for chat: {model_label}. Use /models to choose a chat-capable model."
    if status_code in {401, 403}:
        return "The model provider rejected the credentials. Check the API key and base URL, then retry."
    if status_code == 429:
        return "The model provider rate limit was reached. Wait and retry, or choose another model."
    error_name = type(exc).__name__
    if error_name == "APITimeoutError":
        return "The model provider timed out. Check the provider server and retry."
    if error_name == "APIConnectionError":
        return "Could not connect to the model provider. Check the provider server and base URL, then retry."
    return "The model provider rejected the request. Check the selected model and provider settings, then retry."


def _friendly_searchagent_error_message(exc: SearchAgentError) -> str:
    if str(exc):
        return str(exc)
    return "The run failed inside SearchAgent. Check the configuration and retry."
