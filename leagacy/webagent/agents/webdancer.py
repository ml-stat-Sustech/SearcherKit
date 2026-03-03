from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .base import AgentDecision, AgentState, BaseAgent, Message
from ..tools.base import BaseTool, ToolCall, ToolResult
from ..llm.client import LLMClient
from .prompts import build_webdancer_system_prompt, build_webdancer_user_prompt
from ..tools import build_webdancer_tools

FORMAT_REMINDER = (
    "Please follow the required format.\n"
    "When calling a tool, respond with:\n"
    "<tool_call>\n"
    "{\"name\": \"search or visit\", \"arguments\": {...}}\n"
    "</tool_call>\n"
    "Wrap the final response for the user inside <answer></answer> once you are done."
)

FINAL_ROUND_REMINDER = (
    "You have reached the final reasoning step. Review all observations above and produce your definitive answer "
    "wrapped inside <answer></answer>. Do not call any additional tools."
)


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
        default_tools = list(tools) if tools is not None else None
        super().__init__(tools=default_tools, max_steps=max_steps)
        self.llm = llm
        
    async def init_tools(self):
        tools = (await build_webdancer_tools()).values()
        for tool in tools:
            self.register_tool(tool)  

    # step 1
    def handle_user_message(self, user_input: WebDancerRequest) -> AgentState:
        state = AgentState(user_input=user_input)
        state.messages.append(Message(role="system", content=build_webdancer_system_prompt(self)))
        state.messages.append(Message(role="user", content=build_webdancer_user_prompt(user_input.query)))
        state.scratchpad["query"] = user_input.query
        return state

    # step 2
    async def generate_step_response(self, state: AgentState) -> str:
        if (
            state.steps_taken >= self.max_steps - 1
            # and not state.scratchpad.get("final_round_prompted")
        ):
            state.messages.append(Message(role="user", content=FINAL_ROUND_REMINDER))
            # state.scratchpad["final_round_prompted"] = True

        raw_output = await self.llm.complete(
            self._messages_to_dicts(state.messages), 
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stop=["</tool_call>"])
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
            name, arguments, trimmed = action
            if trimmed and trimmed != latest:
                state.messages[-1].content = trimmed
            if name in self.tools:
                return AgentDecision(kind="tool", tool_call=ToolCall(name=name, arguments=arguments))
            return AgentDecision(kind="continue")

        structured = self._extract_structured_tool_call(latest)
        if structured:
            name, arguments, trimmed_content = structured
            if trimmed_content != latest:
                state.messages[-1].content = trimmed_content
            if name in self.tools:
                return AgentDecision(kind="tool", tool_call=ToolCall(name=name, arguments=arguments))
            return AgentDecision(kind="continue")

        if state.steps_taken > 1 and state.messages and state.messages[-1].role == "assistant":
            last_reminder_step = state.scratchpad.get("last_reminder_step")
            if last_reminder_step != state.steps_taken:
                state.messages.append(Message(role="user", content=FORMAT_REMINDER))
                state.scratchpad["last_reminder_step"] = state.steps_taken

        return AgentDecision(kind="continue")

    # step 5
    def process_tool_result(self, state: AgentState, result: ToolResult) -> Optional[List[object]]:
        output_text = result.output
        response = f"<tool_response>\n{output_text}\n</tool_response>"
        state.messages.append(Message(role="user", content=response))
        return [{"text": response, "append_to_dialogue": False, "stage": "log"}]

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

    def _extract_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, object], str]]:
        match = None
        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
        for candidate in pattern.finditer(content):
            match = candidate
        if match is None:
            return None

        raw_payload = match.group(1).strip()
        cleaned = self._strip_code_fences(raw_payload)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None

        trimmed_content = content[: match.end()].strip()
        return name, arguments, trimmed_content

    def _extract_structured_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, object], str]]:
        start = content.find('{"name"')
        if start == -1:
            return None

        brace_count = 0
        end = None
        in_string = False
        escape_next = False
        for idx in range(start, len(content)):
            char = content[idx]
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = idx + 1
                    break
        if end is None:
            return None

        segment = content[start:end]
        try:
            data = json.loads(segment)
        except json.JSONDecodeError:
            return None

        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None

        trimmed_content = content[:end]
        return name, arguments, trimmed_content

    @staticmethod
    def _strip_code_fences(payload: str) -> str:
        text = payload.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = [line for line in text.splitlines()]
            if len(lines) >= 2:
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
        return text

    def _extract_answer(self, content: str) -> Optional[str]:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        answer = match.group(1).strip()
        return answer or None


    def _normalise_action_blocks(self, content: str) -> str:
        return content.strip()
