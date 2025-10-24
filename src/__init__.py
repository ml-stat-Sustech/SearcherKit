"""Agent_Factory exposes CLI entry points for multiple web agents."""

from .webagent.main import main


def webwalker_main(*args, **kwargs):
    return main(*args, **kwargs)


def webdancer_main(*args, **kwargs):
    return main(*args, **kwargs)


__all__ = ["main", "webwalker_main", "webdancer_main"]
