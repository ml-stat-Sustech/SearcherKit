#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SFT_RUN_CONFIG="${SFT_RUN_CONFIG:-${SCRIPT_DIR}/sft_run_openseeker_bcp.yaml}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  python openseeker_agent/run.py generate
  --config "${SFT_RUN_CONFIG}"
)

"${cmd[@]}"

RUN_JUDGE_AFTER_GENERATE="${RUN_JUDGE_AFTER_GENERATE:-}"
if [[ -z "${RUN_JUDGE_AFTER_GENERATE}" ]]; then
  RUN_JUDGE_AFTER_GENERATE="$(python - "${SFT_RUN_CONFIG}" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

raw = OmegaConf.to_container(OmegaConf.load(Path(sys.argv[1])), resolve=True)
value = ((raw or {}).get("judge") or {}).get("run_after_generate", False)
print(str(value).lower())
PY
)"
fi
if [[ "${RUN_JUDGE_AFTER_GENERATE}" == "1" || "${RUN_JUDGE_AFTER_GENERATE}" == "true" || "${RUN_JUDGE_AFTER_GENERATE}" == "TRUE" ]]; then
  judge_cmd=(
    python openseeker_agent/run.py judge
    --config "${SFT_RUN_CONFIG}"
  )
  "${judge_cmd[@]}"
else
  echo "Judge skipped. Run later:"
  echo "  SCORER_URLS=... SCORER_API_KEY=... SCORER_MODEL_NAME=... python openseeker_agent/run.py judge --config ${SFT_RUN_CONFIG}"
fi
