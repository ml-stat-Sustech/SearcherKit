"""Shared exception taxonomy for searcherkit.

The runtime uses explicit error classes so callers can decide whether an error
is retryable, fatal, configuration-related, or tied to a specific provider.
"""

from __future__ import annotations


class SearcherKitError(Exception):
    """Base class for all project-defined errors."""


class RecoverableError(SearcherKitError):
    """An operation may succeed if retried or resumed later."""


class FatalError(SearcherKitError):
    """An operation cannot continue without configuration or code changes."""


class LLMError(SearcherKitError):
    """Model provider or model-output failure."""
