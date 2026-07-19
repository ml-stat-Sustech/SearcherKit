from __future__ import annotations

import argparse
import asyncio
import copy
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

from searcherkit.common.config import compose_dataclass_config
from searcherkit.common.live_events import LiveEvent
from searcherkit.common.log import disable_console_logging, setup_logger
from searcherkit.llm.base import ClientConfig
from searcherkit.runtime.interactive import InteractiveQueryConfig, InteractiveQueryRunner
from searcherkit.runtime.interactive_selection import discover_model_options
from searcherkit.runtime.runner import RunConfig

from .app import SearcherKitTui

_SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "anthropic", "vllm", "vllm_server"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searcher tui",
        description="Run ad hoc SearcherKit queries in a prompt-toolkit TUI.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Config file path or name under the packaged searcherkit config directory. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set log level to DEBUG for verbose output.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Load LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, and LLM_BASE_URL from this dotenv file.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run one ad hoc query non-interactively through the interactive query runtime.",
    )
    parser.add_argument(
        "--interactive-output-path",
        default=None,
        help="Directory for Interactive Run Records. Defaults to ${output_path}/interactive or outputs/interactive.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides, for example agent.llm_client.model=Qwen3-8B",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv_file(args.env_file)
    setup_logger(level="DEBUG" if args.debug else None)
    _suppress_interactive_warnings()
    config = compose_dataclass_config(
        RunConfig,
        config_path=args.config_path,
        overrides=args.overrides,
    )
    apply_llm_env(config.agent.llm_client)
    interactive_config = to_interactive_config(
        config,
        record_dir=args.interactive_output_path,
    )
    if args.query is not None:
        asyncio.run(run_query_once(config=interactive_config, query=args.query))
        return 0
    discovery_result = asyncio.run(discover_model_options(config.agent.llm_client))
    disable_console_logging()
    app = SearcherKitTui(
        config=interactive_config,
        model_options=discovery_result.options,
        model_discovery_message=discovery_result.message,
    )
    app.run()
    return 0


def to_interactive_config(config: RunConfig, *, record_dir: str | None = None) -> InteractiveQueryConfig:
    agent_config = copy.deepcopy(config.agent)
    agent_config.stream_llm = True
    return InteractiveQueryConfig(
        agent=agent_config,
        output_path=config.output_path,
        logging=config.logging,
        record_dir=record_dir,
    )


async def run_query_once(*, config: InteractiveQueryConfig, query: str) -> Path:
    streaming_line_open = False

    async def print_event(event: LiveEvent) -> None:
        nonlocal streaming_line_open
        if event.kind == "assistant_delta":
            print(event.message, end="", flush=True)
            streaming_line_open = True
            return
        if streaming_line_open:
            print()
            streaming_line_open = False
        print(f"[{event.kind}] {event.message}")

    result = await InteractiveQueryRunner(config=config).run_query(
        query,
        live_event_sink=print_event,
    )
    if streaming_line_open:
        print()
    print(f"Interactive run record: {result.record_path}")
    return result.record_path


def load_dotenv_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_simple_dotenv(env_path)
        return
    load_dotenv(env_path)


def _load_simple_dotenv(path: Path) -> None:
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def apply_llm_env(config: ClientConfig) -> None:
    provider = _clean_env_value(os.getenv("LLM_PROVIDER"))
    model = _clean_env_value(os.getenv("LLM_MODEL"))
    api_key = _clean_env_value(os.getenv("LLM_API_KEY"))
    base_url = _clean_env_value(os.getenv("LLM_BASE_URL"))
    if provider:
        provider_type = provider.strip().lower()
        if provider_type not in _SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                "LLM_PROVIDER must be one of: openai, anthropic, vllm, vllm_server"
            )
        config.type = provider_type
    if model:
        config.model = model

    provider_type = config.type.lower()
    if provider_type not in _SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            "LLM provider must be one of: openai, anthropic, vllm, vllm_server"
        )
    _apply_endpoint_env(config, api_key=api_key, base_url=base_url)


def _apply_endpoint_env(config: Any, *, api_key: str | None, base_url: str | None) -> None:
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'").strip()
    cleaned = "".join(char for char in cleaned if char.isprintable())
    return cleaned or None


def _suppress_interactive_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"\s*'.*' is validated against ConfigStore schema with the same name\.",
        category=UserWarning,
    )
