# SearcherKit SFT

This document describes how to run agentic SFT in this repository and how to
evaluate the trained checkpoint on BrowseComp Plus.

## Environment Boundary

SFT is intentionally isolated from the RL/AReaL training path.

- RL remains under `searcherkit.training.train_dist` and `train/train_dist.yaml`.
- SFT lives under `searcherkit.training.sft` and `train/sft_openseeker_original.yaml`.
- The repository does not vendor or install `ms-swift`.
- Run SFT commands from an environment that already provides `swift` and any
  converter-only dependencies needed by your input format, such as `pyarrow` for
  OpenResearcher parquet files.

## Recommended Route

For reproducing the Search-Agent-SFT route inside this repository, use OpenSeeker data for SFT and evaluate with the same OpenSeeker/SearchVisit protocol on BrowseComp Plus. That means the model learns and is evaluated with:

- `search({"query": ...})`, where `query` may be a string or list of strings.
- `visit({"url": ..., "goal": ...})`, where `url` may be a string or list of strings.

The runtime pieces for this route are isolated from the RL path. The OpenSeeker parser and tool contract are adapted from `Search-Agent-SFT/eval/openseeker`; only the source dispatch and config glue are SearcherKit-specific:

- `searcherkit.llm.parsers.openseeker.OpenSeekerParser` accepts OpenSeeker `<tool_calls_begin>` wrappers.
- `openseeker_search` and `openseeker_visit` wrap existing SearcherKit sources without changing native RL tools.
- `recipe/sft/browsecomp_openseeker.yaml` runs BCP with the OpenSeeker protocol.

## SFT Data Flow

The SFT pipeline has two data formats:

- Canonical SearcherKit SFT JSONL: written to `data.output_path`.
- ms-swift Hermes JSONL: written to `data.train_path` and passed to `swift sft`.

The canonical format keeps the agent trajectory explicit:

```json
{
  "system": "system prompt",
  "tools": [{"type": "function", "function": {"name": "search", "description": "...", "parameters": {}}}],
  "messages": [
    {"role": "user", "content": "Question: ..."},
    {"role": "assistant", "content": "reasoning"},
    {"role": "tool_call", "content": "{\"name\":\"search\",\"arguments\":{\"query\":\"...\"}}"},
    {"role": "tool_response", "content": "..."},
    {"role": "assistant", "content": "<answer>...</answer>"}
  ]
}
```

The ms-swift export preserves those message roles and serializes `tools` as the
JSON string expected by `agent_template=hermes`.

## Supported Inputs

The default `train/sft_openseeker_original.yaml` uses `input_type: existing` and points at the prepared Search-Agent-SFT red-search JSONL. Other supported input types, used by explicit conversion configs such as `train/sft_openseeker_converted.yaml`, are:

- `existing`: validate and export an already converted agentic JSONL file.
- `openresearcher`: read OpenResearcher seed directories with parquet files.
- `openseeker`: read OpenSeeker trajectory JSONL.

Important data fields:

- `data.input_path`: raw input path.
- `data.output_path`: canonical SearcherKit SFT JSONL output.
- `data.train_path`: ms-swift training JSONL output.
- `data.max_tool_response_chars`: truncate long tool responses before SFT.
- `data.drop_repeated_search_turns`: for OpenSeeker, optionally drop repeated
  search turns and their paired tool responses. Keep this `false` when
  reproducing the Search-Agent-SFT conversion.
- `data.system_prompt_path`: optional prompt override file.

## Convert And Inspect

From the repo root, with the SFT environment active:

```bash
python -m searcherkit.training.sft.train --config train/sft_openseeker_converted.yaml --convert-only
```

For raw OpenSeeker conversion, this writes the reusable converted data outside the repository output directory:

```text
/home/jovyan/data/searcherkit/sft/openseeker_canonical.jsonl
/home/jovyan/data/searcherkit/sft/openseeker_ms_swift.jsonl
```

Default red-search training uses the prepared JSONL directly. Training jobs should use `--skip-convert` and avoid mixing conversion into GPU training.

Use dry-run to validate the config and print the exact `swift sft` command
without starting GPU training:

```bash
bash train/sft.sh --dry-run --skip-convert
```

## Train

`train/sft_openseeker_original.yaml` is the default Search-Agent-SFT/OpenSeeker red-search training route. The default backend settings follow the Search-Agent-SFT Qwen3/Hermes
setup:

```yaml
backend:
  swift_bin: swift
  command: sft
  model: /home/jovyan/Qwen3-8B
  template: qwen3
  agent_template: hermes
  loss_scale: hermes
  output_dir: outputs/sft/model
```

Run training:

```bash
bash train/sft.sh --skip-convert
```

The default SFT config reports training metrics to SwanLab:

```yaml
backend:
  report_to: swanlab
  extra_args:
    swanlab_project: ${oc.env:SWANLAB_PROJECT,searcherkit-sft}
    swanlab_exp_name: ${oc.env:SWANLAB_EXP_NAME,openseeker-qwen3-8b}
    swanlab_mode: ${oc.env:SWANLAB_MODE,cloud}
```

Provide `SWANLAB_API_KEY` in the runtime environment, or use an existing SwanLab login. The repository does not store SwanLab tokens.

For kjob training on one node with 8 GPUs:

```bash
./kjob/submit_sft_openseeker.sh
```

To train on OpenSeeker:

```yaml
data:
  input_type: openseeker
  input_path: /path/to/openseeker.jsonl
  output_path: /home/jovyan/data/searcherkit/sft/openseeker_canonical.jsonl
  train_path: /home/jovyan/data/searcherkit/sft/openseeker_ms_swift.jsonl
  drop_repeated_search_turns: false
```

To train on OpenResearcher parquet:

```yaml
data:
  input_type: openresearcher
  input_path: /path/to/OpenResearcherDataset
  seeds: all
```

OpenResearcher expects seed folders like `seed_42/*.parquet`.

## BrowseComp Plus Evaluation

Evaluation is not run inside `swift sft`. For OpenSeeker SFT checkpoints, use the strict OpenSeeker BCP path in this repository:

- Generation: `searcherkit.training.sft.openseeker_bcp.generate`
- Search/visit tools: `searcherkit.training.sft.openseeker_bcp.mcp_server`
- Judge: `searcherkit.training.sft.openseeker_bcp.judge`

This path mirrors `Search-Agent-SFT/eval/openseeker`: it uses `/v1/completions`, the OpenSeeker Jinja chat template, the `<|im_start|>assistant\n<think>\n` generation prompt, `search`/`visit` tool calls through MCP, `result_tool*.jsonl` outputs, and the A/B judge prompt.

### Required Services

BCP eval needs these services:

- SFT checkpoint served by SGLang/vLLM with an OpenAI-compatible `/v1/completions` endpoint.
- BCP Elasticsearch index, currently `http://10.32.33.135:9200`.
- Embedding service compatible with `/v1/embeddings`.
- Summary service compatible with `/v1/chat/completions`.
- Judge service compatible with `/v1/chat/completions` if running judge.

The support-service job prints reusable URLs:

```bash
./kjob/submit_serve_bcp_support.sh
```

Look for:

```text
Embedding base URL: http://<node-ip>:8004/v1
Summary base URL: http://<node-ip>:6010/v1
```

### Run Generation

If the model, embedding, summary, and MCP services are already running, run generation directly:

```bash
export PYTHONPATH=/home/jovyan/code/searcherkit/src:$PYTHONPATH
export OPENSEEKER_BASE_URL=http://127.0.0.1:8001/v1
export OPENSEEKER_MODEL=Qwen3-8B-SearchVisit-SFT
export MCP_ENDPOINT=http://127.0.0.1:8303/mcp/

BCP_DATASET=/home/jovyan/data/searcherkit/browsecomp_plus_decrypted_qa.jsonl \
OUTPUT_DIR=outputs/sft/openseeker_bcp_strict \
bash scripts/sft/evaluate_sft_bcp_openseeker.sh
```

The runner writes:

- `outputs/sft/openseeker_bcp_strict/input.normalized.jsonl`
- `outputs/sft/openseeker_bcp_strict/result_tool200.jsonl`
- `outputs/sft/openseeker_bcp_strict/result_tool200_metrics.json`
- `outputs/sft/openseeker_bcp_strict/result_tool200.run.log`

### Run With kjob

This job requests one node and 8 GPUs for the SFT model server. It starts the in-repo MCP server and reuses external embedding/summary endpoints:

```bash
SFT_CKPT=/home/jovyan/code/searcherkit/outputs/sft/model/v2-20260620-150109/checkpoint-1459 \
MODEL_TOKENIZER_PATH=/data/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
EMBEDDING_BASE_URL=http://<support-node-ip>:8004/v1 \
SUMMARY_BASE_URL=http://<support-node-ip>:6010/v1 \
BCP_ES_HOST=http://10.32.33.135:9200 \
BCP_ES_INDEX=browsecomp_plus_qwen3-embedding-8b \
OUTPUT_DIR=outputs/sft/openseeker_bcp_strict_ckpt1459 \
./kjob/submit_eval_sft_bcp_openseeker_strict.sh
```

### Judge Saved Outputs

Run the OpenSeeker A/B judge over `result_tool200.jsonl`:

```bash
SCORER_URLS=http://<judge-node-ip>:8000/v1 \
SCORER_API_KEY=EMPTY \
SCORER_MODEL_NAME=/data/hf/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
python -m searcherkit.training.sft.openseeker_bcp.judge \
  --data_path outputs/sft/openseeker_bcp_strict_ckpt1459/result_tool200.jsonl \
  --max_workers 20
```

The judge writes `result_tool200_eval.jsonl`. Its first line is a summary with `accuracy`, `correct_num`, `wrong_num`, `unknown_num`, and tool-call statistics.

## Checks

Useful smoke checks:

```bash
python -m searcherkit.training.sft.train --config train/sft_openseeker_original.yaml --dry-run --max-records 1
python -m searcherkit.training.sft.train --config train/sft_openseeker_converted.yaml --convert-only --max-records 1
python -m pytest tests/test_training_sft.py
```

If `openresearcher` conversion fails with a missing `pyarrow` import, install it
in the SFT conversion environment rather than changing the repository runtime
environment.
