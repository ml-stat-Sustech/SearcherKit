import os


if os.environ.get("AREAL_ENABLE_STEP_LEVEL_CLIPPING") == "1":
    import patch_areal_functional

    patch_areal_functional.install()
