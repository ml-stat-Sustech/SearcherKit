from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..llm.client import LLMClient, OpenAIChatClient
from ..agents.prompts import STSTEM_CRITIIC_ANSWER, STSTEM_CRITIIC_INFORMATION


@dataclass
class MemoryManager:
    """Encapsulates two-stage memory extraction and critique workflow."""

    llm: LLMClient
    query: str
    judge_llm: Optional[LLMClient] = field(default=None, repr=False)
    max_retries: int = 3
    backoff_base: float = 1.0
    entries: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure the judge client defaults to the agent LLM when unspecified."""

        if self.judge_llm is None:
            self.judge_llm = self.llm

    def reset(self, query: Optional[str] = None) -> None:
        """Clear accumulated memory and optionally reset the tracked query."""

        if query is not None:
            self.query = query
        self.entries.clear()

    def configure_judge_llm(
        self,
        llm: Optional[LLMClient] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Override the LLM client used for critique/judge calls.

        If neither a client nor model is provided, the agent's LLM is reused.
        """

        if llm and model:
            raise ValueError("Provide either an LLM client instance or a model name, not both.")
        if model:
            self.judge_llm = OpenAIChatClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
                default_kwargs=default_kwargs,
            )
            return
        if llm:
            self.judge_llm = llm
            return
        self.judge_llm = self.llm

    def _run_json_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        client: Optional[LLMClient] = None,
    ) -> Optional[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        delay = self.backoff_base
        llm_client = client or self.llm
        for attempt in range(self.max_retries):
            try:
                raw = llm_client.complete(messages, response_format={"type": "json_object"})
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
            except Exception as exc:  # noqa: BLE001 - bubble up with retries
                if attempt == self.max_retries - 1:
                    raise exc
                time.sleep(delay)
                delay *= 2
        return None

    def analyze_observation(
        self,
        observation: str,
    ) -> Tuple[Optional[bool], Optional[str], Optional[Dict[str, Any]]]:
        """
        Evaluate whether an observation is useful and extract a memory snippet if so.

        Returns
        -------
        Tuple[Optional[bool], Optional[str], Optional[Dict[str, Any]]]
            (useful flag, extracted memory, raw payload). The useful flag is ``None`` when
            the LLM response could not be parsed.
        """

        user_prompt = f"- Query: {self.query}\n- Observation: {observation}"
        payload = self._run_json_completion(STSTEM_CRITIIC_INFORMATION, user_prompt)
        if not payload:
            return None, None, None
        usefulness = payload.get("usefulness")
        if isinstance(usefulness, str):
            usefulness = usefulness.lower() == "true"
        if usefulness is None:
            return None, None, payload
        if usefulness:
            information = payload.get("information")
            if isinstance(information, str):
                self.entries.append(information.strip())
                return True, information.strip(), payload
            return True, None, payload
        return False, None, payload

    def judge_completion(self) -> Tuple[Optional[bool], Optional[str], Optional[Dict[str, Any]]]:
        """
        Decide whether the accumulated memory is sufficient to answer the query.

        Returns
        -------
        Tuple[Optional[bool], Optional[str], Optional[Dict[str, Any]]]
            (judge flag, answer text, raw payload). The flag is ``None`` when judging
            could not be performed.
        """

        if not self.entries:
            return None, None, None
        joined = "-".join(self.entries)
        user_prompt = f"- Query: {self.query}\n- Accumulated Information: {joined}"
        payload = self._run_json_completion(STSTEM_CRITIIC_ANSWER, user_prompt, client=self.judge_llm)
        if not payload:
            return None, None, None
        judge = payload.get("judge")
        if isinstance(judge, str):
            judge = judge.lower() == "true"
        if judge is None:
            return None, None, payload
        if judge:
            answer = payload.get("answer")
            if isinstance(answer, str):
                return True, answer.strip(), payload
            return True, None, payload
        return False, None, payload
