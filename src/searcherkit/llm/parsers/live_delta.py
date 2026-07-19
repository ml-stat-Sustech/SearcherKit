"""Reusable streaming interpreters for parser-specific output protocols."""

from __future__ import annotations

from searcherkit.llm.parsers.base import LiveDeltaPart


class AnswerTagLiveDeltaSplitter:
    """Interpret ``<answer>`` boundaries in provider-native streaming text."""

    _TAGS = ("<answer>", "</answer>")

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "content"

    def feed(self, text: str) -> list[LiveDeltaPart]:
        self._buffer += text
        out: list[LiveDeltaPart] = []
        while self._buffer:
            tag_start = self._buffer.find("<")
            if tag_start > 0:
                self._emit(out, self._buffer[:tag_start])
                self._buffer = self._buffer[tag_start:]
                continue
            if tag_start == -1:
                self._emit(out, self._buffer)
                self._buffer = ""
                break

            matched = False
            for tag in self._TAGS:
                if self._buffer.startswith(tag):
                    if tag == "<answer>":
                        self._mode = "final_answer"
                        out.append(LiveDeltaPart(field="final_answer", text=""))
                    else:
                        self._mode = "suppressed"
                    self._buffer = self._buffer[len(tag) :]
                    matched = True
                    break
            if matched:
                continue
            if any(tag.startswith(self._buffer) for tag in self._TAGS):
                break
            self._emit(out, "<")
            self._buffer = self._buffer[1:]
        return out

    def flush(self) -> list[LiveDeltaPart]:
        if not self._buffer:
            return []
        out: list[LiveDeltaPart] = []
        self._emit(out, self._buffer)
        self._buffer = ""
        return out

    def _emit(self, out: list[LiveDeltaPart], text: str) -> None:
        if not text or self._mode == "suppressed":
            return
        field = "final_answer" if self._mode == "final_answer" else "content"
        if out and out[-1].field == field:
            out[-1] = LiveDeltaPart(field=field, text=out[-1].text + text)
            return
        out.append(LiveDeltaPart(field=field, text=text))
