from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


LiveEventKind = Literal[
    "run_started",
    "run_completed",
    "run_failed",
    "run_cancelled",
    "user_message",
    "assistant_turn_started",
    "assistant_delta",
    "assistant_message",
    "tool_call_started",
    "tool_result",
]


@dataclass(slots=True)
class LiveEvent:
    kind: LiveEventKind
    message: str
    data: dict[str, Any] = field(default_factory=dict)


LiveEventSink = Callable[[LiveEvent], Awaitable[None] | None]


async def emit_live_event(sink: LiveEventSink | None, event: LiveEvent) -> None:
    if sink is None:
        return
    result = sink(event)
    if inspect.isawaitable(result):
        await result


def _event_payload(event: LiveEvent) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "kind": event.kind,
        "message": event.message,
        "data": event.data,
    }


class RunEventRecorder:
    """Records events for later persistence."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: LiveEvent) -> None:
        if event.kind == "assistant_delta":
            return
        self.events.append(_event_payload(event))


class MultiSink:
    """Forwards one event to multiple sinks."""

    def __init__(self, *sinks: LiveEventSink) -> None:
        self.sinks = sinks

    async def __call__(self, event: LiveEvent) -> None:
        for sink in self.sinks:
            await emit_live_event(sink, event)
