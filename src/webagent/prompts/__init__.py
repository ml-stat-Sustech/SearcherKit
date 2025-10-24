"""Prompt helpers for different agents."""

from .webwalker import (
    STSTEM_CRITIIC_ANSWER,
    STSTEM_CRITIIC_INFORMATION,
    SYSTEM_EXPLORER,
)
from .webdancer import build_system_prompt as build_webdancer_system_prompt
from .webdancer import build_user_prompt as build_webdancer_user_prompt

__all__ = [
    "SYSTEM_EXPLORER",
    "STSTEM_CRITIIC_INFORMATION",
    "STSTEM_CRITIIC_ANSWER",
    "build_webdancer_system_prompt",
    "build_webdancer_user_prompt",
]
