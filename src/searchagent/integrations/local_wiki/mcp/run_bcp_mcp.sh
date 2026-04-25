#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(cd "${REPO_ROOT}/.."; pwd)"

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-/home/jovyan/Qwen3-Embedding-8B}"
export EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-http://127.0.0.1:8004/v1}"

export SUMMARY_MODEL="${SUMMARY_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-8B-FP8}"
export SUMMARY_BASE_URL="${LOCAL_WIKI_SUMMARY_BASE_URL:-http://192.168.77.17:8300/v1}"
export SUMMARY_API_KEY="${LOCAL_WIKI_SUMMARY_API_KEY:-EMPTY}"

export ELASTICSEARCH_HOST="http://10.32.36.206:9200"
export ELASTICSEARCH_INDEX="browsecomp_plus_qwen3-embedding-8b"

uvicorn searchagent.integrations.local_wiki.mcp.bcp_mcp:create_app --factory --host 0.0.0.0 --port 8303 --workers 1
