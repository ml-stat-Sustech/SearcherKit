"""Shared exception taxonomy for searchagent.

The runtime uses explicit error classes so callers can decide whether an error
is retryable, fatal, configuration-related, or tied to a specific provider.
"""

from __future__ import annotations


class SearchAgentError(Exception):
    """Base class for all project-defined errors."""


class RecoverableError(SearchAgentError):
    """An operation may succeed if retried or resumed later."""


class FatalError(SearchAgentError):
    """An operation cannot continue without configuration or code changes."""


class ConfigError(FatalError):
    """Configuration is missing, invalid, or points to an unknown component."""


class LLMError(SearchAgentError):
    """Model provider or model-output failure."""


class SourceError(SearchAgentError):
    """Data source or document retrieval failure."""



