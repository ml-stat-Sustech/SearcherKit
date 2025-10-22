"""Agent_Factory exposes CLI entry points for multiple web agents."""

def main(*args, **kwargs):
    from .main import main as _main
    return _main(*args, **kwargs)


def webwalker_main(*args, **kwargs):
    return main(*args, **kwargs)


def webdancer_main(*args, **kwargs):
    return main(*args, **kwargs)


__all__ = ["main", "webwalker_main", "webdancer_main"]
