from __future__ import annotations

import os


if os.environ.get("AREAL_ENABLE_STEP_LEVEL_CLIPPING") == "1":
    try:
        import patch_areal_functional
    except ModuleNotFoundError as exc:
        if exc.name not in {"areal", "patch_areal_functional", "torch"}:
            raise
    else:
        patch_areal_functional.install()
