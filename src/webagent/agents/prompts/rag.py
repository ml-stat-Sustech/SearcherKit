"""Prompt helpers for the single-round RAG agent."""

from __future__ import annotations

from datetime import datetime


def build_rag_system_prompt() -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return (
        "You are a retrieval-augmented assistant that responds in English. "
        "Always gather supporting evidence with the `search` tool before attempting to answer. "
        "After receiving search results (and any follow-up visits), synthesise a concise response that cites the most relevant facts. "
        "Always prefix the final reply with `Final Answer:`.\n\n"
        f"Current datetime: {now}."
    )


def build_rag_user_prompt(query: str) -> str:
    return (
        "Question: {query}\n"
        "Call the `search` tool first to retrieve information relevant to this question. "
        "After you receive the search results (and any subsequent observations), combine the evidence and produce a response in English with the `Final Answer:` prefix."
    ).format(query=query.strip())


def build_rag_context_prompt(query: str, search_result: str) -> str:
    return (
        "Question: {query}\n"
        "The following material was retrieved from search/visit tools:\n"
        "```\n"
        "{search_result}\n"
        "```\n"
        "Summarise the key evidence and provide a final answer in English, prefixed with `Final Answer:`."
    ).format(query=query.strip(), search_result=search_result.strip())
