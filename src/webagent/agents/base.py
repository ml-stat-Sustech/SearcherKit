from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, AsyncIterator, List, Literal, Optional

from ..tools.base import BaseTool, ToolCall, ToolResult


@dataclass
class Message:
    """Lightweight chat message representation used by the base agent."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentDecision:
    """Decision returned by `decide_next_action`."""

    kind: Literal["continue", "tool", "final"]
    message: Optional[str] = None
    tool_call: Optional[ToolCall] = None


@dataclass
class AgentEvent:
    """Streamed event emitted during `BaseAgent.run`."""

    stage: Literal[
        "receive",
        "initial_response",
        "decision",
        "tool_result",
        "response",
        "final",
        "log"
    ]
    payload: Any


@dataclass
class AgentState:
    """Mutable execution state shared between workflow stages."""

    user_input: Any
    messages: List[Message] = field(default_factory=list)
    scratchpad: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[ToolResult] = field(default_factory=list)
    steps_taken: int = 0


class BaseAgent(abc.ABC):
    """
    Base implementation of the six-stage agent workflow.

    Subclasses implement the abstract hooks to provide model-specific logic.
    The `run` method orchestrates the standard loop and streams `AgentEvent`s.
    """

    def __init__(
        self,
        *,
        tools: Optional[List[BaseTool]] = None,
        max_steps: int = 20,
    ) -> None:
        self.max_steps = max_steps
        self.tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, tool: BaseTool) -> None:
        """Register or replace a tool implementation."""

        self.tools[tool.name] = tool

    async def run(self, user_input: Any) -> AsyncIterator[AgentEvent]:
        """Execute the agent workflow and yield streaming events."""

        # step 1
        state = self.handle_user_message(user_input)
        yield AgentEvent(stage="receive", payload=state.messages)
        yield AgentEvent(stage="log", payload="Prepare User Message")

        while state.steps_taken < self.max_steps:
            state.steps_taken += 1

            # step 2
            if state.steps_taken > 1:
                yield AgentEvent(stage="log", payload="================== Response =====================")
                assistant_reply = await self.generate_step_response(state)
                if assistant_reply:
                    state.messages.append(Message(role="assistant", content=assistant_reply))
                    yield AgentEvent(stage="response", payload=assistant_reply)
                    pending_final = state.scratchpad.pop("final_answer", None)
                    if pending_final:
                        state.messages.append(Message(role="assistant", content=pending_final))
                        yield AgentEvent(stage="final", payload=pending_final)
                        break

            # step 3
            yield AgentEvent(stage="log", payload="================== Decision =====================")

            decision = self.decide_next_action(state)
            yield AgentEvent(stage="decision", payload=decision)

            if decision.kind == "final":
                final_answer = self.finalize_response(state, decision)
                yield AgentEvent(stage="final", payload=final_answer)
                break

            # step 4
            yield AgentEvent(stage="log", payload="================== Leveraging Tools =====================")

            if decision.kind == "tool":
                if decision.tool_call is None:
                    raise ValueError("Tool decision missing ToolCall details.")
                result = await self.execute_tool(decision.tool_call, state)
                state.tool_results.append(result)
                yield AgentEvent(stage="tool_result", payload=result)

                # step 5
                yield AgentEvent(stage="log", payload="================== Processing Tool Results =====================")

                follow_ups = self.process_tool_result(state, result)

                if follow_ups:
                    for entry in follow_ups:
                        if isinstance(entry, dict):
                            text = str(entry.get("text", "")).strip()
                            append_to_dialogue = bool(entry.get("append_to_dialogue", True))
                            stage = str(entry.get("stage", "response"))
                        else:
                            text = str(entry).strip()
                            append_to_dialogue = True
                            stage = "response"
                        if not text:
                            continue
                        if append_to_dialogue:
                            state.messages.append(Message(role="assistant", content=text))
                        # yield AgentEvent(stage=stage, payload=text)
                pending_final = state.scratchpad.pop("final_answer", None)
                if pending_final:
                    state.messages.append(Message(role="assistant", content=pending_final))
                    yield AgentEvent(stage="final", payload=pending_final)
                    break
                continue

            if decision.kind == "continue":
                continue

            raise ValueError(f"Unsupported decision kind: {decision.kind}")
        else:
            fallback = self.on_max_steps(state)
            yield AgentEvent(stage="final", payload=fallback)

    async def execute_tool(self, call: ToolCall, state: AgentState) -> ToolResult:
        """Execute the requested tool using the registry."""

        if call.name not in self.tools:
            raise KeyError(f"Tool '{call.name}' is not registered.")
        output = await self.tools[call.name].run(call, state)
        return ToolResult(call=call, output=output)

    def on_max_steps(self, state: AgentState) -> str:
        """Fallback final answer when the step budget is exhausted."""

        return "Reached the maximum number of steps without producing a final answer."

    @abc.abstractmethod
    def handle_user_message(self, user_input: Any) -> AgentState:
        """Stage 1: receive and normalize the user request."""

    @abc.abstractmethod
    async def generate_step_response(self, state: AgentState) -> str:
        """Stage 2+: call the LLM to extend the assistant side of the dialogue."""

    @abc.abstractmethod
    def decide_next_action(self, state: AgentState) -> AgentDecision:
        """Stage 3: decide whether to invoke a tool or reply directly."""

    @abc.abstractmethod
    def process_tool_result(self, state: AgentState, result: ToolResult) -> Optional[List[object]]:
        """
        Stage 5: incorporate the tool output back into the conversation.

        Return a list of follow-up payloads (strings or dictionaries) to emit.
        """

    @abc.abstractmethod
    def finalize_response(self, state: AgentState, decision: AgentDecision) -> str:
        """Stage 6: produce the final answer once the loop terminates."""
