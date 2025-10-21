from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .base import AgentDecision, AgentState, BaseAgent, Message
from ..tools.base import BaseTool, ToolCall, ToolResult
from ..llm.client import LLMClient
from ..prompts import build_webdancer_system_prompt, build_webdancer_user_prompt
from ..tools import WEBDANCER_TOOLS


@dataclass
class WebDancerRequest:
    query: str


class WebDancerAgent(BaseAgent):
    def __init__(
        self,
        llm: LLMClient,
        *,
        tools: Optional[Iterable[BaseTool]] = None,
        max_steps: int = 20,
    ) -> None:
        super().__init__(tools=list(tools or WEBDANCER_TOOLS.values()), max_steps=max_steps)
        self.llm = llm

    # step 1
    def handle_user_message(self, user_input: WebDancerRequest) -> AgentState:
        state = AgentState(user_input=user_input)
        state.messages.append(Message(role="system", content=build_webdancer_system_prompt()))
        state.messages.append(Message(role="user", content=build_webdancer_user_prompt(user_input.query)))
        state.scratchpad["query"] = user_input.query
        return state

    # step 2
    def generate_step_response(self, state: AgentState) -> str:
        return self.llm.complete(self._messages_to_dicts(state.messages))

    # step 3
    def decide_next_action(self, state: AgentState) -> AgentDecision:
        if not state.messages or state.messages[-1].role != "assistant":
            return AgentDecision(kind="continue")

        latest = state.messages[-1].content
        answer = self._extract_answer(latest)
        if answer:
            return AgentDecision(kind="final", message=answer)

        action = self._extract_tool_call(latest)
        if action:
            name, arguments = action
            if name in self.tools:
                return AgentDecision(kind="tool", tool_call=ToolCall(name=name, arguments=arguments))
        return AgentDecision(kind="continue")

    # step 5
    def process_tool_result(self, state: AgentState, result: ToolResult) -> Optional[List[str]]:
        response = f"<tool_response>\n{result.output}\n</tool_response>"
        state.messages.append(Message(role="user", content=response))
        return None

    def finalize_response(self, state: AgentState, decision: AgentDecision) -> str:
        if decision.message:
            return decision.message.strip()
        if state.messages and state.messages[-1].role == "assistant":
            fallback = self._extract_answer(state.messages[-1].content)
            if fallback:
                return fallback
            return state.messages[-1].content.strip()
        return "Unable to provide an answer, please try again.。"

    def _messages_to_dicts(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _extract_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, object]]]:
        match = re.search(r"<tool_call>\s*(.+?)\s*</tool_call>", content, re.DOTALL)
        if not match:
            return None
        payload = match.group(1).strip()
        try:
            data = json.loads(payload)
            name = data.get("name")
            arguments = data.get("arguments", {})
        except json.JSONDecodeError:
            return None
        if isinstance(name, str) and isinstance(arguments, dict):
            return name, arguments
        return None

    def _extract_answer(self, content: str) -> Optional[str]:
        match = re.search(r"<answer>\s*(.+?)\s*</answer>", content, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()
