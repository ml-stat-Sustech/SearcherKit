from __future__ import annotations

import asyncio
import sys

try:
    import uvloop
except ImportError:
    uvloop = None

if uvloop is not None:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from searchagent.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
