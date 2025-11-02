"""Prompt helpers for the Vanilla agent."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an assistant. "
    "Please Answer questions directly using your own knowledge and logical deduction. "
    "Do not mention any tool usage because you cannot access external tools. "
    "Always provide short reasoning if needed, then conclude with a line that starts with 'Final Answer: '."
)


def build_vanilla_system_prompt() -> str:
    """Return the static system prompt for the Vanilla agent."""

    return SYSTEM_PROMPT


def build_vanilla_user_prompt(query: str) -> str:
    """Format the initial user prompt for the Vanilla agent."""

    return f"Question:\n{query}\n\nThought: "
