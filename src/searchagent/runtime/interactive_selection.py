"""Interactive selection state and helpers.

Selection State scopes future Interactive Query Runs. It is shared by terminal UI
code and the interactive runtime, but it is not terminal-display state and it is
not part of a completed run by itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from searchagent.agent.search_agent import SearchAgentConfig
from searchagent.llm.base import (
    ClientConfig,
    OpenAIConfig,
    VllmConfig,
)
from searchagent.llm.base import OpenAIConfig as OpenAICompatibleProviderConfig
from searchagent.sources import SourceConfig

SOURCE_BACKED_TOOL_TYPES = frozenset({"search", "visit"})
SAFE_SOURCE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SOURCE_COMMAND_RE = re.compile(
    r"^sources:(?P<name>[A-Za-z0-9_.-]+)\((?P<source_type>[A-Za-z0-9_.-]+)\)$"
)
_OPENAI_COMPATIBLE_DISCOVERY_PROVIDERS = {"openai", "vllm", "vllm_server"}

ActiveSourceKind = Literal["active", "mixed", "none", "missing"]


@dataclass(frozen=True, slots=True)
class SourceOption:
    """A source that can be selected for future Interactive Query Runs."""

    name: str
    source_type: str

    @property
    def command_name(self) -> str:
        return f"sources:{self.name}({self.source_type})"

    @property
    def label(self) -> str:
        return f"{self.name}({self.source_type})"


@dataclass(frozen=True, slots=True)
class ActiveSource:
    """The Active Source state inferred from source-backed tool configuration."""

    kind: ActiveSourceKind
    name: str | None = None
    source_type: str | None = None
    tool_sources: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        if self.kind == "active" and self.name and self.source_type:
            return f"{self.name}({self.source_type})"
        if self.kind == "mixed":
            return f"mixed:{','.join(self.tool_sources)}" if self.tool_sources else "mixed"
        if self.kind == "missing" and self.name:
            return f"missing:{self.name}"
        return "none"

    def as_event_data(self) -> dict[str, object]:
        payload: dict[str, object] = {"state": self.kind}
        if self.name is not None:
            payload["name"] = self.name
        if self.source_type is not None:
            payload["type"] = self.source_type
        if self.tool_sources:
            payload["tool_sources"] = list(self.tool_sources)
        return payload


@dataclass(frozen=True, slots=True)
class ModelOption:
    """A model target selectable for future Interactive Query Runs."""

    provider: str
    model: str
    base_url: str | None

    @property
    def command_name(self) -> str:
        return f"models:{self.provider}/{self.model}"

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True, slots=True)
class ModelDiscoveryResult:
    options: list[ModelOption]
    failed: bool = False
    message: str = ""


@dataclass
class SelectionState:
    """Session-scoped selection of Active Model and Active Source.

    This state scopes future Interactive Query Runs. It is not part of any
    completed run and may be shared by the terminal UI, an API, or any other
    interactive client.
    """

    active_model: ModelOption | None = None
    active_source: ActiveSource | None = None


# --- Active Source ---------------------------------------------------------


def is_safe_source_token(value: str) -> bool:
    """Return whether *value* can appear unescaped in a source slash command."""

    return bool(SAFE_SOURCE_TOKEN_RE.fullmatch(value))


def selectable_source_options(sources: list[SourceConfig]) -> list[SourceOption]:
    """Return slash-command-safe source options in config order."""

    options: list[SourceOption] = []
    for source in sources:
        name = str(source.name or "")
        source_type = str(source.type or "")
        if not is_safe_source_token(name) or not is_safe_source_token(source_type):
            continue
        options.append(SourceOption(name=name, source_type=source_type))
    return options


def source_option_by_name(sources: list[SourceConfig], name: str) -> SourceOption | None:
    """Find a configured source by name, including sources unsafe for commands."""

    for source in sources:
        if source.name != name:
            continue
        source_type = str(source.type or "")
        return SourceOption(name=str(source.name), source_type=source_type)
    return None


def parse_source_command(command: str, sources: list[SourceConfig]) -> SourceOption | None:
    """Parse `/sources:name(type)` without the leading slash."""

    match = SOURCE_COMMAND_RE.fullmatch(command)
    if match is None:
        return None
    name = match.group("name")
    source_type = match.group("source_type")
    for option in selectable_source_options(sources):
        if option.name == name and option.source_type == source_type:
            return option
    return None


def active_source_label(agent_config: SearchAgentConfig, active_source: ActiveSource | None) -> str:
    """Return the human-readable label for the active or inferred source."""

    if active_source is not None:
        return active_source.label
    return infer_active_source(agent_config).label


def infer_active_source(agent_config: SearchAgentConfig) -> ActiveSource:
    """Infer the effective Active Source from built-in source-backed tools."""

    source_types = {source.name: source.type for source in agent_config.sources if source.name}
    tool_source_names: list[str] = []
    has_multi_source_binding = False
    for tool in agent_config.tools:
        if str(tool.type or "") not in SOURCE_BACKED_TOOL_TYPES:
            continue
        if not tool.source:
            continue
        source_names = [str(name) for name in tool.source if name]
        has_multi_source_binding = has_multi_source_binding or len(source_names) > 1
        tool_source_names.extend(source_names)

    unique_sources = tuple(sorted(set(tool_source_names)))
    if not unique_sources:
        return ActiveSource(kind="none")
    if has_multi_source_binding or len(unique_sources) > 1:
        return ActiveSource(kind="mixed", tool_sources=unique_sources)

    name = unique_sources[0]
    source_type = source_types.get(name)
    if source_type is None:
        return ActiveSource(kind="missing", name=name, tool_sources=unique_sources)
    return ActiveSource(kind="active", name=name, source_type=str(source_type), tool_sources=unique_sources)


def active_source_for_name(agent_config: SearchAgentConfig, name: str) -> ActiveSource:
    """Build an Active Source state for a selected configured source name."""

    option = source_option_by_name(agent_config.sources, name)
    if option is None:
        return ActiveSource(kind="missing", name=name, tool_sources=(name,))
    return ActiveSource(
        kind="active",
        name=option.name,
        source_type=option.source_type,
        tool_sources=(option.name,),
    )


def apply_active_source(agent_config: SearchAgentConfig, source_name: str) -> int:
    """Point built-in source-backed tools at *source_name*.

    Returns the number of tool configs updated.
    """

    updated = 0
    for tool in agent_config.tools:
        if str(tool.type or "") not in SOURCE_BACKED_TOOL_TYPES:
            continue
        tool.source = [source_name]
        updated += 1
    return updated


# --- Active Model ----------------------------------------------------------


def openai_compatible_provider_config(config: ClientConfig) -> OpenAICompatibleProviderConfig | None:
    provider = str(config.type or "").lower()
    if provider == "openai":
        return config.openai
    if provider in {"vllm", "vllm_server"}:
        return config.vllm
    return None


def _base_url_values(base_url: Any | None) -> list[str | None]:
    if base_url is None or isinstance(base_url, str):
        return [base_url]
    if isinstance(base_url, Sequence) and not isinstance(base_url, (str, bytes, bytearray)):
        return [str(item) if item is not None else None for item in base_url]
    return [str(base_url)]


def _safe_discovery_base_url(base_url: str | None) -> str:
    return base_url or "default OpenAI endpoint"


async def discover_model_options(config: ClientConfig) -> ModelDiscoveryResult:
    provider = str(config.type or "").lower()
    if provider not in _OPENAI_COMPATIBLE_DISCOVERY_PROVIDERS:
        return ModelDiscoveryResult(options=[])

    provider_config = openai_compatible_provider_config(config)
    if provider_config is None:
        return ModelDiscoveryResult(
            options=[],
            failed=True,
            message=f"Model discovery failed for {provider}: provider config is missing.",
        )

    options: list[ModelOption] = []
    seen: set[str] = set()
    failures: list[str] = []
    for base_url in _base_url_values(provider_config.base_url):
        try:
            discovered = await _discover_model_options_for_endpoint(
                provider=provider,
                base_url=base_url,
                api_key=provider_config.api_key,
                extra_client_kwargs=provider_config.extra_client_kwargs,
            )
        except (ImportError, OSError, TimeoutError, ValueError) as exc:
            failures.append(f"{_safe_discovery_base_url(base_url)} ({exc})")
            continue
        for option in discovered:
            if option.command_name in seen:
                continue
            seen.add(option.command_name)
            options.append(option)

    if options:
        return ModelDiscoveryResult(options=options)
    if failures:
        return ModelDiscoveryResult(
            options=[],
            failed=True,
            message=f"Model discovery failed for {provider}: {', '.join(failures)}.",
        )
    return ModelDiscoveryResult(options=[])


async def _discover_model_options_for_endpoint(
    *,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    extra_client_kwargs: dict[str, Any] | None,
) -> list[ModelOption]:
    from openai import APIConnectionError, APIError, APITimeoutError, OpenAIError
    from openai import AsyncOpenAI

    client_kwargs = dict(extra_client_kwargs or {})
    client_kwargs.setdefault("timeout", 5.0)
    try:
        client = AsyncOpenAI(api_key=api_key or "ollama", base_url=base_url, **client_kwargs)
        response = await client.models.list()
    except (APIConnectionError, APITimeoutError, APIError, OpenAIError) as exc:
        raise ValueError(str(exc)) from exc

    response_data = getattr(response, "data", None)
    if response_data is None:
        raise ValueError("models response missing data")

    options: list[ModelOption] = []
    for model in response_data:
        model_id = getattr(model, "id", None)
        if not model_id:
            continue
        options.append(ModelOption(provider=provider, model=str(model_id), base_url=base_url))
    return options


def parse_model_command(command: str, options: Sequence[ModelOption]) -> ModelOption | None:
    for option in options:
        if command == option.command_name:
            return option
    return None


def active_model_label(config: ClientConfig, active_model: ModelOption | None) -> str:
    """Return the human-readable label for the active or configured model."""

    if active_model is not None:
        return active_model.label
    provider = str(config.type or "provider")
    model = str(config.model or "model")
    return f"{provider}/{model}"


def active_model_event_data(config: ClientConfig) -> dict[str, Any]:
    """Return non-secret Active Model metadata for an Interactive Query Run event."""

    provider = str(config.type or "")
    provider_cfg = openai_compatible_provider_config(config)
    if provider_cfg is None and provider.lower() == "anthropic":
        provider_cfg = config.anthropic
    return {
        "provider": provider,
        "model": config.model,
        "base_url": getattr(provider_cfg, "base_url", None) if provider_cfg is not None else None,
    }


def apply_active_model(config: ClientConfig, option: ModelOption) -> None:
    config.type = option.provider
    config.model = option.model
    provider_config = openai_compatible_provider_config(config)
    if provider_config is None:
        provider_config = _ensure_openai_compatible_provider_config(config, option.provider)
    provider_config.base_url = option.base_url


def _ensure_openai_compatible_provider_config(
    config: ClientConfig,
    provider: str,
) -> OpenAICompatibleProviderConfig:
    if provider == "openai":
        config.openai = config.openai or OpenAIConfig()
        return config.openai
    if provider in {"vllm", "vllm_server"}:
        config.vllm = config.vllm or VllmConfig()
        return config.vllm
    raise ValueError(f"Unsupported model option provider: {provider}")
