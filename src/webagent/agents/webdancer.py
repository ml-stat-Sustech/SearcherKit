from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .base import AgentDecision, AgentState, BaseAgent, Message
from ..tools.base import BaseTool, ToolCall, ToolResult
from ..llm.client import LLMClient
from ..prompts import build_webdancer_system_prompt, build_webdancer_user_prompt
from ..tools import build_webdancer_tools


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
        default_tools = list(tools) if tools is not None else list(build_webdancer_tools().values())
        super().__init__(tools=default_tools, max_steps=max_steps)
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
        raw_output = self.llm.complete(self._messages_to_dicts(state.messages))
        return self._normalise_action_blocks(raw_output)

    # step 3
    def decide_next_action(self, state: AgentState) -> AgentDecision:
        if state.steps_taken > 1:
            if not state.messages or state.messages[-1].role != "assistant":
                return AgentDecision(kind="continue", message=None)

        latest = state.messages[-1].content
        if state.steps_taken > 1:

            answer = self._extract_answer(latest)
            if answer:
                return AgentDecision(kind="final", message=answer)

        action = self._extract_tool_call(latest)
        if action:
            name, arguments = action
            if name in self.tools:
                return AgentDecision(kind="tool", tool_call=ToolCall(name=name, arguments=arguments))
            raise KeyError(f"Tool '{name}' is not registered. Available tools: {', '.join(self.tools.keys())}")
        return AgentDecision(kind="continue")

    # step 5
    def process_tool_result(self, state: AgentState, result: ToolResult) -> Optional[List[str]]:
        response = f"Observation:\n{result.output}\nThought:"
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
        action_name: Optional[str] = None
        argument_lines: List[str] = []
        capturing_arguments = False

        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            lowered = stripped.lower()

            if lowered.startswith("action:"):
                action_name = raw_line.split(":", 1)[1].strip()
                if action_name.startswith(("\"", "'")) and action_name.endswith(("\"", "'")):
                    action_name = action_name[1:-1].strip()
                capturing_arguments = False
                argument_lines = []
                continue

            if lowered.startswith("action input:"):
                remainder = raw_line.split(":", 1)[1].strip()
                capturing_arguments = True
                argument_lines = [remainder] if remainder else []
                continue

            if capturing_arguments:
                if not stripped:
                    if argument_lines:
                        break
                    continue

                if any(
                    lowered.startswith(prefix)
                    for prefix in ("thought:", "action:", "observation:", "final answer:", "decision:")
                ):
                    break

                decision_prefix = stripped.lstrip("=🧭- ")
                if decision_prefix.lower().startswith("decision"):
                    break

                argument_lines.append(stripped)

        if not action_name:
            return None

        arguments_raw = "\n".join(argument_lines).strip()
        if arguments_raw.startswith("```"):
            fence_lines = [line.strip() for line in arguments_raw.splitlines()]
            fence_lines = fence_lines[1:]
            if fence_lines and fence_lines[-1].startswith("```"):
                fence_lines = fence_lines[:-1]
            arguments_raw = "\n".join(fence_lines).strip()

        if not arguments_raw:
            arguments: Dict[str, object] = {}
        else:
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError:
                return None

        if not isinstance(arguments, dict):
            return None

        return action_name, arguments

    def _extract_answer(self, content: str) -> Optional[str]:
        marker = re.search(r"Final Answer:\s*", content, re.IGNORECASE)
        print(marker)
        if not marker:
            return None

        answer = content[marker.end() :].strip()
        return answer or None


    def _normalise_action_blocks(self, content: str) -> str:
        """
        Ensure each action input line terminates before any observation text.

        Some model responses append 'Observation:' directly after the action input.
        Splitting them keeps the environment-generated observation separate from the
        tool invocation issued by the agent.
        """

        prefix_lines: List[str] = []
        action_lines: List[str] = []
        captured_primary_action = False
        captured_action_input = False

        for line in content.splitlines():
            stripped = line.lstrip()
            lower = stripped.lower()

            if lower.startswith("observation:"):
                continue

            if lower.startswith("action:"):
                if captured_primary_action:
                    break
                captured_primary_action = True
                action_lines.append(line)
                continue

            if captured_primary_action:
                if not captured_action_input:
                    action_lines.append(line)
                continue

            prefix_lines.append(line)

        combined = prefix_lines + action_lines
        return "\n".join(combined).strip()
