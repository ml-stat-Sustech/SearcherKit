#!/bin/bash

set -euo pipefail

export OPENAI_BASE_URL="https://www.dmxapi.cn/v1"
export OPENAI_API_KEY="REMOVED_REVOKED_SECRET"

exec uv run python -m searchagent evaluate \
    outputs/gemma4_12b \
    outputs/gemma4_12b/eval \
    --max-concurrency 128
