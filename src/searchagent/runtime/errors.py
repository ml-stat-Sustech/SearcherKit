from __future__ import annotations

from searchagent.errors import FatalError, RecoverableError, SearchAgentError


class RuntimeErrorBase(SearchAgentError):
    """Base class for runtime orchestration failures."""


class RuntimeConfigError(FatalError):
    """Runtime configuration is invalid or incomplete."""


class CheckpointError(RuntimeErrorBase):
    """Checkpoint state cannot be read, written, or interpreted safely."""


class CheckpointCorruptionError(CheckpointError):
    """Checkpoint payload is not valid JSON or has an unexpected shape."""


class ResumeStateError(RecoverableError):
    """A run can continue from persisted checkpoint state."""

