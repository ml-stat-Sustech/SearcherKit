#!/bin/bash

set -euo pipefail

export OPENAI_BASE_URL="https://www.dmxapi.cn/v1"
export OPENAI_API_KEY=""

searcher evaluate \
    /path/to/input \
    /path/to/output \
    --max-concurrency 128 \
    --answer-pattern '\\boxed\{(?P<answer>[^}]*)\}'
