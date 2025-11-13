# WebAgent

`WebAgent` is a reusable multi-agent CLI framework that currently ships **WebWalker**, **WebDancer**, a single-round **RAG** agent, and a lightweight **Vanilla** baseline. The shared abstraction lives in `src/webagent/agents/base.py`, which implements the six-stage workflow (receive request → initial reasoning → action selection → tool execution → observation handling → final answer). Each concrete agent only needs to provide its prompt, tools, and a small amount of agent-specific logic.

Directory overview:

- `src/webagent/agents/`: base class plus concrete agents (`webwalker.py`, `webdancer.py`, `rag.py`, `vanilla.py`)
- `src/webagent/common/`: factory helpers (`builder.py`), shared state (`state.py`), reusable utilities (`utils.py`), and the shared memory manager
- `src/webagent/prompts/`: prompt templates per agent
- `src/webagent/tools/`: reusable tool implementations (WebWalker’s `visit_page`; WebDancer’s `search`/`visit`)
- `src/webagent/cli.py`: unified CLI entry that selects agents via command-line flags

## Setup

```bash
conda create -n agent-factory python=3.10
conda activate agent-factory
pip install -r requirements.txt
crawl4ai-setup   # one-time crawler initialization
```

If you plan to use the local wiki toolchain, see `src/local_wiki/README.md` for extra dependencies (Elasticsearch, embedding models, etc.).

## LLM configuration

By default the CLI looks for an OpenAI-compatible endpoint via environment variables:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o           # or any chat-capable model ID
# Optional:
export OPENAI_MODEL_SERVER=https://your.server/v1
export OPENAI_TEMPERATURE=0.3
export OPENAI_MAX_OUTPUT_TOKENS=800
```

### Tool configuration

- `GOOGLE_SEARCH_KEY`: Serper.dev API key for batched search (default cloud workflow).
- `JINA_API_KEY`: used by the visit tool to fetch webpage content (default cloud workflow).
- `DASHSCOPE_API_KEY` (and optional `DASHSCOPE_MODEL_SERVER` / `WEBDANCER_VISIT_MODEL`): summarisation backend for `visit` (default cloud workflow).

#### Using the local wiki pipeline (local_wiki)

You can replace the default web search/visit tools with the local Wikipedia retrievers provided by the `src/local_wiki/` package:

1. Prepare the index following `src/local_wiki/README.md` (launch Elasticsearch, build an index with `eswiki/wiki2index_links.py`, etc.).
2. Install dependencies from the repo root: `pip install -r requirements.txt`.
3. Export the environment variables:
   ```bash
   export WEBDANCER_USE_LOCAL_WIKI=1
   export LOCAL_WIKI_INDEX=wiki20251001_qwen3-embedding-0.6b   # replace with your index name
   export LOCAL_WIKI_ES_HOST=http://192.168.77.12:9200             # optional; defaults to this value
   export LOCAL_WIKI_RETRIEVER=dense                            # bm25, dense, or hybrid
   export LOCAL_WIKI_MODEL_NAME=/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B/   # required for dense/hybrid
   # Optional agent-specific toggle when using the single-round RAG agent:
   export RAG_USE_LOCAL_WIKI=1
   # Optional overrides:
   # export LOCAL_WIKI_SEARCH_TOP_K=10
   # export LOCAL_WIKI_MAX_LINKS=50
   # export LOCAL_WIKI_BODY_MAX_CHARS=6000
   ```
4. Run WebDancer as usual:
   ```bash
   python -m src.webagent.demo --agent webdancer --query "维基百科中周杰伦的音乐奖项有哪些？"
   ```

With `WEBDANCER_USE_LOCAL_WIKI=1`, the agent’s `search` tool becomes a local index lookup (returning page titles), and the `visit` tool fetches article bodies plus actionable intra-wiki links via Elasticsearch instead of hitting the public web. Clear or unset the flag to revert to the online Serper/Jina/DashScope workflow.
You can also toggle the same environment variables directly from the dataset runner with `--use-local-wiki-tools`, `--local-wiki-index`, and related flags (see “Dataset sweeps” below).

### Using a local OpenAI-compatible server (e.g., vLLM)

You can host Qwen or any other model locally and expose it via the OpenAI API schema. For example, with [vLLM](https://github.com/vllm-project/vllm):

```bash
pip install vllm
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
 --model /mnt/sharedata/ssd_large/common/LLMs/Qwen/WebDancer-32B/ \
 --port 8000 \
 --dtype float16 \
 --tensor-parallel-size 4 \
 --gpu-memory-utilization 0.8
```

Then configure the CLI to hit the local endpoint:

```bash
export OPENAI_MODEL=/mnt/sharedata/ssd_large/common/LLMs/Qwen/WebDancer-32B/
export OPENAI_MODEL_SERVER=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-demo  # any non-empty string
```

Now run single queries via the demo runner:

```bash
python -m src.webagent.demo --agent webwalker \
  --website https://example.com \
  --query "Find the latest announcements."

python -m src.webagent.demo --agent webdancer \
  --query "查找ACL 2025行业轨道的截稿时间和会场地址"

python -m src.webagent.demo --agent rag \
  --query "概述最新的星舰试飞进展，并列出两个主要信息来源"

# 使用本地 Wiki 检索运行 RAG：
RAG_USE_LOCAL_WIKI=1 \
LOCAL_WIKI_INDEX=wiki20251001_qwen3-embedding-0.6b \
python -m src.webagent.demo \
  --agent rag \
  --query "简要介绍三体问题在经典力学中的意义"
```

Setting `RAG_USE_LOCAL_WIKI=1` switches the RAG agent to the local search/visit toolchain (shared with WebDancer).

This workflow also works with other OpenAI-compatible gateways (FastChat, text-generation-inference, llama.cpp server, LMDeploy, etc.).

## Usage

### Unified CLI

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m src.webagent.cli \
  --agent webwalker \
  --website https://2025.aclweb.org/ \
  --query "When is the paper submission deadline for ACL 2025 Industry Track?"
  --max-rounds 30	

python -m src.webagent.cli \
  --agent webdancer \
  --query "Find the submission deadline and venue address for the ACL 2025 industry track." \
  --max-rounds 30
```

Flags:

- `--agent` selects `webwalker`, `webdancer`, `rag`, or `vanilla` (default `webwalker`).
- `--website` is required when `--agent webwalker` is chosen.
- `--max-rounds` controls the step budget (defaults: 10 for WebWalker, 20 for WebDancer, 8 for RAG, 4 for Vanilla if omitted).

Ensure the additional tool-related environment variables are present when running WebDancer.

### Dataset sweeps (GAIA, WebWalkerQA, custom JSONL)

The dataset runner lives in `src/webagent/main.py` and accepts either a Hugging Face dataset identifier or a local directory/file path:

- Hugging Face hub datasets (e.g. `callanwu/WebWalkerQA`) are loaded via `datasets.load_dataset`.
- Local GAIA checkouts or metadata files are detected automatically and parsed without needing the deprecated GAIA dataset script.
- Plain JSON/JSONL files containing `question`/`answer` style records are also supported.

Key flags:

- `--dataset-name`: Hugging Face name or local path (`./GAIA`, `/data/custom.jsonl`, etc.).
- `--dataset-split`: split name (`train`, `validation`, `test`, `main`, …). For GAIA directories, this matches the folder containing `metadata.jsonl`.
- `--use-local-wiki-tools` and related flags mirror the environment variables described earlier, letting you toggle local search per run.
- `--run-eval` enables inline LLM judge scoring (see below).

Example GAIA + local wiki run:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m src.webagent.main \
  --agent webdancer \
  --dataset-name ./GAIA \
  --dataset-split validation \
  --output-path runs/gaia_webdancer.jsonl \
  --max-samples 5 \
  --use-local-wiki-tools \
  --local-wiki-index wiki20251001_qwen3-embedding-0.6b \
  --local-wiki-es-host http://127.0.0.1:9200 \
  --local-wiki-retriever dense \
  --local-wiki-model-name /path/to/embedding/model \
  --run-eval
```

Switching datasets is just a matter of pointing `--dataset-name` at the desired source (e.g. `callanwu/WebWalkerQA`, `./GAIA`, `./my_experiments/questions.jsonl`) and adjusting `--dataset-split` accordingly. The loader normalises common field names (`Question`, `Final answer`, `Annotator Metadata`, etc.), so mixed datasets can share the same runner.

The output JSONL contains the prediction (`pred`), the original answer (when available), and a `metadata` block with GAIA identifiers (task ID, level, attachment filenames). Set `--run-eval` to invoke the LLM judge immediately after inference.

## Evaluation

WebAgent provides two interchangeable workflows for LLM-judge evaluation of prediction files.

### Option 1: Evaluate during inference

Invoke `src/main.py` with the new evaluation flags. The CLI first writes predictions to `--output-path`, then immediately runs the judge and generates both a scored JSONL file and a summary report.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -m src.webagent.main \
  --agent webwalker \
  --dataset-name callanwu/WebWalkerQA \
  --dataset-split main \
  --output-path /tmp/webwalker_predictions.jsonl \
  --max-rounds 15 \
  --run-eval \
  --eval-output-path /tmp/webwalker_predictions_scored.jsonl \
  --force-rejudge
```

Flags of note:

- `--run-eval` enables evaluation once inference completes.
- `--eval-output-path` (optional) controls where the scored JSONL is written. Default: append `_eval` to `--output-path`.
- `--force-rejudge` forces the judge to rescore questions even if results already exist in the output file. All evaluations use the WebWalkerQA-style prompt by default.

The helper script `run_main.sh` now uses this integrated path by default—adjust the environment variables at the top of the script and run `bash run_main.sh`.

### Option 2: Evaluate an existing predictions file

You can still call the standalone evaluator on any JSONL produced earlier (e.g., by the unified CLI or from archived runs):

```bash
python -m src.webagent.evaluate.evl \
  --input_path /mnt/sharedata/hdd/beier/Agent/WebWalker/webwalker_predictions.jsonl \
  --output_path /mnt/sharedata/hdd/beier/Agent/WebWalker/webwalker_predictions_scored.jsonl \
  --force-rejudge
```

Both workflows create:

- A scored JSONL (`*_scored.jsonl` / `*_eval.jsonl`) containing per-question scores and judge metadata.
- A companion summary report (`*_report.json`) with aggregate accuracy broken down by difficulty and source type.

Make sure the relevant LLM judge environment variables (OpenAI-compatible endpoint, keys, etc.) are configured before running either evaluation flow.

### Running a separate judge model with vLLM

You can dedicate a different GPU pool/model to the judge while the agent continues to use its own deployment. Run two OpenAI-compatible vLLM servers (one per model) and wire them into the agent factory:

```bash
# Agent model on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /mnt/sharedata/ssd_large/common/LLMs/Qwen/WebDancer-32B/ \
   --dtype float16 \
 --tensor-parallel-size 4 \
 --gpu-memory-utilization 0.8 \
  --port 8000

# Judge model on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
  --model /mnt/sharedata/ssd_large/common/LLMs/Qwen/Qwen2.5-14B-Instruct/ \
  --tensor-parallel-size 2 \
  --port 8001
```

For the dataset runner (`src/webagent/main.py`), export the relevant environment variables and add `--use-separate-judge-llm`:

```bash
export OPENAI_MODEL=/mnt/sharedata/ssd_large/common/LLMs/Qwen/WebDancer-32B/
export OPENAI_MODEL_SERVER=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-demo

export OPENAI_JUDGE_MODEL=/mnt/sharedata/ssd_large/common/LLMs/Qwen/Qwen2.5-7B-Instruct/
export OPENAI_JUDGE_MODEL_SERVER=http://127.0.0.1:8001/v1
export OPENAI_JUDGE_API_KEY=local-demo

python -m src.webagent.main \
  --agent webwalker \
  --dataset-name ... \
  --output-path ... \
  --use-separate-judge-llm

# Separate judge LLM for standalone evaluator
python -m src.webagent.evaluate.evl \
  --input_path ... \
  --output_path ... \
  --use-separate-judge-llm
```

If you already have a context, call `context.agent.memory.configure_judge_llm(...)` with either a pre-built `LLMClient` or a `model` + `base_url` pair to rebind only the judge. When no override is provided, the judge automatically falls back to the agent’s LLM.

## Creating a custom agent

To add your own agent on top of this framework:

1. **Define tools (optional)**  
   - Implement new `BaseTool` subclasses in `src/webagent/tools/`, or reuse existing ones.  
   - Register them in `src/webagent/tools/__init__.py`, e.g. `MYAGENT_TOOLS`.

2. **Prepare prompts**  
   - Add a prompt module in `src/webagent/prompts/` that builds system/user prompts for the new agent.

3. **Implement the agent**  
   - Create `src/webagent/agents/myagent.py`, inherit `BaseAgent`, and implement `handle_user_message`, `generate_step_response`, `decide_next_action`, `process_tool_result`, `finalize_response`.  
   - Export the request dataclass and agent via `src/webagent/agents/__init__.py`.

4. **Register with the factory**  
   - Update `src/webagent/common/builder.py:create_agent` to route a new `--agent` value and return `AgentRunContext(agent=my_agent, request=...)`.

5. **Invoke from CLI**  
   - Run `python -m src.webagent.cli --agent myagent ...`. The unified CLI now streams the new agent’s reasoning/actions without further plumbing.

Both CLIs stream each workflow stage—prompts, model thoughts, tool invocations, tool responses, and final answers—directly to stdout. Interrupt safely with `Ctrl+C`.
