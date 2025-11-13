from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..agents.base import BaseAgent
from ..tools.base import BaseTool
from ..llm.client import LLMClient, build_llm_from_env

if TYPE_CHECKING:  # pragma: no cover - only for static type checkers
    from ..agents.webwalker import WebWalkerAgent, WebWalkerRequest
    from ..agents.webdancer import WebDancerAgent, WebDancerRequest
    from ..agents.vanilla import VanillaAgent, VanillaRequest
    from ..agents.rag import RAGAgent, RAGRequest


@dataclass
class AgentRunContext:
    """Bundle containing a ready-to-run agent and its initial request payload."""

    agent: BaseAgent
    request: object


def build_webwalker_agent(
    *,
    website: str,
    query: str,
    max_rounds: int = 10,
    llm: Optional[LLMClient] = None,
    tools: Optional[list[BaseTool]] = None,
    judge_llm: Optional[LLMClient] = None,
) -> AgentRunContext:
    """
    Convenience helper used by the CLI.

    This wires together the LLM client, WebWalker agent, and request payload while
    keeping the default tool registry configurable for downstream callers.
    """

    from ..agents.webwalker import WebWalkerAgent, WebWalkerRequest
    from ..tools import build_webwalker_tools

    llm_client = llm or build_llm_from_env()
    tool_list = tools or list(build_webwalker_tools().values())
    agent = WebWalkerAgent(llm=llm_client, tools=tool_list, max_steps=max_rounds, judge_llm=judge_llm)
    request = WebWalkerRequest(website=website, query=query)
    return AgentRunContext(agent=agent, request=request)


def build_webdancer_agent(
    *,
    query: str,
    max_rounds: int = 20,
    llm: Optional[LLMClient] = None,
    tools: Optional[list[BaseTool]] = None,
) -> AgentRunContext:
    from ..agents.webdancer import WebDancerAgent, WebDancerRequest
    from ..tools import build_webdancer_tools

    llm_client = llm or build_llm_from_env()
    tool_list = tools or list(build_webdancer_tools().values())
    agent = WebDancerAgent(llm=llm_client, tools=tool_list, max_steps=max_rounds)
    request = WebDancerRequest(query=query)
    return AgentRunContext(agent=agent, request=request)


def build_rag_agent(
    *,
    query: str,
    max_rounds: int = 8,
    llm: Optional[LLMClient] = None,
    tools: Optional[list[BaseTool]] = None,
) -> AgentRunContext:
    from ..agents.rag import RAGAgent, RAGRequest
    from ..tools import build_rag_tools

    llm_client = llm or build_llm_from_env()
    tool_list = tools or list(build_rag_tools().values())
    agent = RAGAgent(llm=llm_client, tools=tool_list, max_steps=max_rounds)
    request = RAGRequest(query=query)
    return AgentRunContext(agent=agent, request=request)


def build_vanilla_agent(
    *,
    query: str,
    max_rounds: int = 4,
    llm: Optional[LLMClient] = None,
) -> AgentRunContext:
    from ..agents.vanilla import VanillaAgent, VanillaRequest

    llm_client = llm or build_llm_from_env()
    agent = VanillaAgent(llm=llm_client, max_steps=max_rounds)
    request = VanillaRequest(query=query)
    return AgentRunContext(agent=agent, request=request)


def create_agent(
    agent_name: str,
    *,
    query: str,
    website: Optional[str] = None,
    max_rounds: Optional[int] = None,
    llm: Optional[LLMClient] = None,
    tools: Optional[list[BaseTool]] = None,
    judge_llm: Optional[LLMClient] = None,
) -> AgentRunContext:
    """
    Factory helper that instantiates the requested agent.

    Parameters
    ----------
    agent_name:
        Identifier of the agent to build. Supported values: ``webwalker``, ``webdancer``, ``rag``, ``vanilla``.
    query:
        User query to investigate.
    website:
        Root website for agents that need a starting URL (required for WebWalker).
    max_rounds:
        Override the default step budget for the selected agent.
    llm/tools:
        Optional overrides for the language model client or tool list.
    judge_llm:
        Optional override for the critiquing/judge LLM. Defaults to the agent LLM.
    """

    name = agent_name.lower()
    if name == "webwalker":
        if not website:
            raise ValueError("webwalker agent requires a --website argument.")
        return build_webwalker_agent(
            website=website,
            query=query,
            max_rounds=max_rounds or 10,
            llm=llm,
            tools=tools,
            judge_llm=judge_llm,
        )

    if name == "webdancer":
        return build_webdancer_agent(
            query=query,
            max_rounds=max_rounds or 20,
            llm=llm,
            tools=tools,
        )

    if name == "rag":
        return build_rag_agent(
            query=query,
            max_rounds=max_rounds or 3,
            llm=llm,
            tools=tools,
        )

    if name == "vanilla":
        return build_vanilla_agent(
            query=query,
            max_rounds=max_rounds or 4,
            llm=llm,
        )

    raise ValueError(
        f"Unsupported agent '{agent_name}'. Available agents: webwalker, webdancer, rag, vanilla."
    )
