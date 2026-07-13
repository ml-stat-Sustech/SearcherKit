"""Dataclasses for chat viewport text selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatPoint:
    row: int
    col: int


@dataclass(slots=True)
class ChatSelection:
    anchor: ChatPoint
    head: ChatPoint
