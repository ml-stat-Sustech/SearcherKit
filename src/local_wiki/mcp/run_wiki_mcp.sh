SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(cd "${REPO_ROOT}/.."; pwd)"

export CUDA_VISIBLE_DEVICES=0
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B}"
export VLLM_PORT="${VLLM_PORT:-8200}"

uvicorn src.local_wiki.mcp:app --host 0.0.0.0 --port 8100 --workers 4