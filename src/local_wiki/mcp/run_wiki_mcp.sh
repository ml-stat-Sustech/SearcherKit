SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(cd "${REPO_ROOT}/.."; pwd)"

export CUDA_VISIBLE_DEVICES=0
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B}"
export VLLM_PORT="${VLLM_PORT:-8200}"

export SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-8B}"
export SUMMARY_MODEL_PORT="${SUMMARY_MODEL_PORT:-8300}"
export LOCAL_WIKI_SUMMARY_MODEL="${LOCAL_WIKI_SUMMARY_MODEL:-Qwen3-8B}"
export LOCAL_WIKI_SUMMARY_BASE_URL="${LOCAL_WIKI_SUMMARY_BASE_URL:-http://localhost:${SUMMARY_MODEL_PORT}/v1}"
export LOCAL_WIKI_SUMMARY_API_KEY="${LOCAL_WIKI_SUMMARY_API_KEY:-EMPTY}"

uvicorn src.local_wiki.mcp:app --host 0.0.0.0 --port 8100 --workers 4