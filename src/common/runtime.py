from __future__ import annotations

import os
import json
from typing import Callable, Dict, IO, Optional, TYPE_CHECKING

from .builder import AgentRunContext, create_agent

if TYPE_CHECKING:  # pragma: no cover
    from ..agents.base import AgentDecision, AgentEvent, Message
    from ..llm import LLMClient
    from ..tools.base import ToolResult


def emit(text: str = "", *, end: str = "\n", log_handle: Optional[IO[str]] = None) -> None:
    """Print text to stdout and mirror it into the optional log file handle."""
    print(text, end=end)
    if log_handle:
        log_handle.write(text)
        log_handle.write(end)
        log_handle.flush()


def _format_arguments(arguments: Optional[dict]) -> str:
    if not arguments:
        return "{}"
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except TypeError:
        return repr(arguments)


def render_event(event: "AgentEvent", *, emit_func: Callable[[str], None]) -> None:
    """
    Render key details for each agent event when verbose logging is enabled.

    Shows incoming messages, assistant replies, tool usage, and final answers.
    """
    stage = event.stage

    if stage == "receive":
        message: "Message" = event.payload
        emit_func(f"👤 {message.role.capitalize()} message:")
        emit_func(message.content)
        return

    if stage == "response":
        emit_func("🤖 Assistant response:")
        emit_func(str(event.payload))
        return

    if stage == "decision":
        decision: "AgentDecision" = event.payload
        if decision.kind == "tool" and decision.tool_call:
            emit_func(
                f"🧭 Decision: call tool '{decision.tool_call.name}' "
                f"with args {_format_arguments(decision.tool_call.arguments)}"
            )
        elif decision.kind == "final":
            emit_func("🧭 Decision: produce final answer.")
        else:
            emit_func(f"🧭 Decision: {decision.kind}")
        return

    if stage == "tool_result":
        result: "ToolResult" = event.payload
        emit_func(f"🛠 Tool '{result.call.name}' output:")
        emit_func(result.output)
        return

    if stage == "final":
        emit_func("🎯 Final Answer:")
        emit_func(str(event.payload))
        return

    emit_func(f"ℹ️ Event [{stage}]: {event.payload}")


def _build_judge_llm_from_env() -> "LLMClient":
    """Construct a judge-specific LLM client using OPENAI_JUDGE_* environment variables."""

    from ..llm.client import OpenAIChatClient  # Lazy import to avoid circular deps

    model = os.environ.get("OPENAI_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError(
            "No judge model configured. Set OPENAI_JUDGE_MODEL (or OPENAI_MODEL) before enabling "
            "the separate judge LLM."
        )
    base_url = os.environ.get("OPENAI_JUDGE_MODEL_SERVER") or os.environ.get("OPENAI_MODEL_SERVER")
    api_key = os.environ.get("OPENAI_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    default_kwargs: Dict[str, object] = {}
    temperature = os.environ.get("OPENAI_JUDGE_TEMPERATURE")
    if temperature:
        default_kwargs["temperature"] = float(temperature)
    max_tokens = os.environ.get("OPENAI_JUDGE_MAX_OUTPUT_TOKENS")
    if max_tokens:
        default_kwargs["max_tokens"] = int(max_tokens)

    return OpenAIChatClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        default_kwargs=default_kwargs or None,
    )


def build_context(
    args,
    *,
    query: str,
    website: Optional[str],
    llm_client: Optional["LLMClient"] = None,
    judge_llm_client: Optional["LLMClient"] = None,
) -> AgentRunContext:
    """Instantiate the requested agent with the provided query and website."""
    return create_agent(
        args.agent,
        query=query,
        website=website,
        max_rounds=args.max_rounds,
        llm=llm_client,
        judge_llm=judge_llm_client,
    )


def ensure_parent_directory(path: str) -> None:
    """Create parent directories for the given path if necessary."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def run_single_query(
    args,
    *,
    query: str,
    website: Optional[str],
    emit_func: Callable[[str], None],
    verbose: bool = True,
) -> str:
    """Execute one agent run and return the final answer."""
    judge_llm_client = None
    if getattr(args, "use_separate_judge_llm", False):
        judge_llm_client = _build_judge_llm_from_env()

    context = build_context(
        args,
        query=query,
        website=website,
        judge_llm_client=judge_llm_client,
    )

    if not getattr(args, "_printed_llm_info", False):
        agent_llm = getattr(context.agent, "llm", None)
        agent_name = None
        if agent_llm:
            agent_name = agent_llm.name() if hasattr(agent_llm, "name") else agent_llm.__class__.__name__
        memory = getattr(context.agent, "memory", None)
        judge_llm = getattr(memory, "judge_llm", None) if memory else None
        judge_name = None
        if judge_llm:
            judge_name = judge_llm.name() if hasattr(judge_llm, "name") else judge_llm.__class__.__name__
        if agent_name:
            emit_func(f"🤖 Agent LLM: {agent_name}")
        if judge_name:
            emit_func(f"⚖️ Judge LLM: {judge_name}")
        setattr(args, "_printed_llm_info", True)

    final_answer = ""

    for event in context.agent.run(context.request):
        if verbose:
            render_event(event, emit_func=emit_func)
        if event.stage == "final":
            final_answer = event.payload

    return final_answer
