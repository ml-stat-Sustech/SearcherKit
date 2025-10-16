from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient
from ..prompts import STSTEM_CRITIIC_ANSWER, STSTEM_CRITIIC_INFORMATION


@dataclass
class MemoryManager:
    """Encapsulates two-stage memory extraction and critique workflow."""

    llm: LLMClient
    query: str
    max_retries: int = 3
    backoff_base: float = 1.0
    entries: List[str] = field(default_factory=list)

    def reset(self, query: Optional[str] = None) -> None:
        """Clear accumulated memory and optionally reset the tracked query."""

        if query is not None:
            self.query = query
        self.entries.clear()

    def _run_json_completion(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        delay = self.backoff_base
        for attempt in range(self.max_retries):
            try:
                raw = self.llm.complete(messages, response_format={"type": "json_object"})
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
            except Exception as exc:  # noqa: BLE001 - bubble up with retries
                if attempt == self.max_retries - 1:
                    raise exc
                time.sleep(delay)
                delay *= 2
        return None

    def analyze_observation(self, observation: str) -> Optional[str]:
        """Return extracted info if observation is deemed useful."""

        user_prompt = f"- Query: {self.query}\n- Observation: {observation}"
        payload = self._run_json_completion(STSTEM_CRITIIC_INFORMATION, user_prompt)
        if not payload:
            return None
        usefulness = payload.get("usefulness")
        if isinstance(usefulness, str):
            usefulness = usefulness.lower() == "true"
        if usefulness:
            information = payload.get("information")
            if isinstance(information, str):
                self.entries.append(information.strip())
                return information.strip()
        return None

    def judge_completion(self) -> Optional[str]:
        """Return final answer if accumulated memory suffices."""

        if not self.entries:
            return None
        joined = "-".join(self.entries)
        user_prompt = f"- Query: {self.query}\n- Accumulated Information: {joined}"
        payload = self._run_json_completion(STSTEM_CRITIIC_ANSWER, user_prompt)
        if not payload:
            return None
        judge = payload.get("judge")
        if isinstance(judge, str):
            judge = judge.lower() == "true"
        if judge:
            answer = payload.get("answer")
            if isinstance(answer, str):
                return answer.strip()
        return None
