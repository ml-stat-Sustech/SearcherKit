SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_TOKEN=${HF_TOKEN:-"REMOVED_REVOKED_SECRET"}
export HF_HOME=${HF_HOME:-/mnt/sharedata/ssd_large/common/LLMs/}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/sharedata/ssd_large/common/datasets/}

# VisitForLocalWiki 
# python "${PROJECT_ROOT}/tools/visit.py"

# # SearchForLocalWiki
python "${PROJECT_ROOT}/tools/search.py"
