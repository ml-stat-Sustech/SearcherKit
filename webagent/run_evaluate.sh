#!/bin/bash

set -euo pipefail

export OPENAI_BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="a"

exec uv run python -m webagent evaluate \
    /path/to/input \
    /path/to/output \
    --max-concurrency 128
