#!/bin/bash
deactivate

source /mnt/sharedata/ssd_large/users/hyli/searchagent/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(cd "${REPO_ROOT}/.."; pwd)"

export VLLM_MODEL="${VLLM_MODEL:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B}"
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://192.168.77.15:8200/v1}"

export SUMMARY_MODEL="${SUMMARY_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-8B-FP8}"
export SUMMARY_BASE_URL="${LOCAL_WIKI_SUMMARY_BASE_URL:-http://192.168.77.17:8300/v1}"
export SUMMARY_API_KEY="${LOCAL_WIKI_SUMMARY_API_KEY:-EMPTY}"

export ELASTICSEARCH_HOST="http://10.32.36.206:9200"
export ELASTICSEARCH_INDEX="wiki20251001_qwen3-embedding-0.6b"

uvicorn searchagent.integrations.local_wiki.mcp.local_wiki_mcp:create_app --factory --host 0.0.0.0 --port 8100 --workers 8
