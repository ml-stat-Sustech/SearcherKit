"""Public package exports for webagent."""

from .log import get_logger, logger, setup_logger


def main(*args, **kwargs):
    from .__main__ import main as run_main

    return run_main(*args, **kwargs)


def webwalker_main(*args, **kwargs):
    return main(*args, **kwargs)


def webdancer_main(*args, **kwargs):
    return main(*args, **kwargs)


def rag_main(*args, **kwargs):
    return main(*args, **kwargs)


__all__ = [
    "get_logger",
    "logger",
    "main",
    "rag_main",
    "setup_logger",
    "webdancer_main",
    "webwalker_main",
]
