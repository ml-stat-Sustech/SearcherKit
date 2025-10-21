"""Agent_Factory exposes CLI entry points for multiple web agents."""

from .main import main

webwalker_main = main
webdancer_main = main

__all__ = ["main", "webwalker_main", "webdancer_main"]
