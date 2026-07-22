from __future__ import annotations

from enum import Enum


class TerminationReason(str, Enum):
    NORMAL = "normal"
    BAD_LAST_TURN = "bad_last_turn"
