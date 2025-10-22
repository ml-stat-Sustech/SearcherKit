# WebAgent

`WebAgent` is a reusable multi-agent CLI framework that currently ships **WebWalker** and **WebDancer** web agents. The shared abstraction lives in `src/agents/base.py`, which implements the six-stage workflow (receive request → initial reasoning → action selection → tool execution → observation handling → final answer). Each concrete agent only needs to provide its prompt, tools, and a small amount of agent-specific logic.

Directory overview:

- `src/agents/`: base class plus concrete agents (`webwalker.py`, `webdancer.py`)
- `src/common/`: factory helpers (`builder.py`), shared state (`state.py`), reusable utilities (`utils.py`), and the shared memory manager
- `src/prompts/`: prompt templates per agent
- `src/tools/`: reusable tool implementations (WebWalker’s `visit_page`; WebDancer’s `search`/`visit`)
- `src/cli.py`: unified CLI entry that selects agents via command-line flags

## Setup

```bash
conda create -n agent-factory python=3.10
conda activate agent-factory
pip install -r requirements.txt
crawl4ai-setup   # one-time crawler initialization
```

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

Additional environment variables for WebDancer tools:

- `GOOGLE_SEARCH_KEY`: Serper.dev API key for batched search.
- `JINA_API_KEY`: used by the visit tool to fetch webpage content.
- `DASHSCOPE_API_KEY` (and optional `DASHSCOPE_MODEL_SERVER` / `WEBDANCER_VISIT_MODEL`): summarisation backend for `visit`.

### Using a local OpenAI-compatible server (e.g., vLLM)

You can host Qwen or any other model locally and expose it via the OpenAI API schema. For example, with [vLLM](https://github.com/vllm-project/vllm):

```bash
pip install vllm
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
 --model /mnt/sharedata/ssd_large/common/LLMs/Qwen/Qwen2.5-32B-Instruct/ \
 --port 8000 \
 --dtype float16 \
 --tensor-parallel-size 4 \
 --gpu-memory-utilization 0.8
```

Then configure the CLI to hit the local endpoint:

```bash
export OPENAI_MODEL=/mnt/sharedata/ssd_large/common/LLMs/Qwen/Qwen2.5-32B-Instruct/
export OPENAI_MODEL_SERVER=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-demo  # any non-empty string
```

Now run single queries via the demo runner:

```bash
python -m webagent.src.demo --agent webwalker \
  --website https://example.com \
  --query "Find the latest announcements."

python -m webagent.src.demo --agent webdancer \
  --query "查找ACL 2025行业轨道的截稿时间和会场地址"
```

This workflow also works with other OpenAI-compatible gateways (FastChat, text-generation-inference, llama.cpp server, LMDeploy, etc.).

## Usage

### Unified CLI

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m webagent.src.cli \
  --agent webwalker \
  --website https://2025.aclweb.org/ \
  --query "When is the paper submission deadline for ACL 2025 Industry Track?"
  --max-rounds 30	

python -m webagent.src.cli \
  --agent webdancer \
  --query "Find the submission deadline and venue address for the ACL 2025 industry track." \
  --max-rounds 30
```

Flags:

- `--agent` selects either `webwalker` or `webdancer` (default `webwalker`).
- `--website` is required when `--agent webwalker` is chosen.
- `--max-rounds` controls the step budget (defaults: 10 for WebWalker, 20 for WebDancer if omitted).

Ensure the additional tool-related environment variables are present when running WebDancer.

## Evaluation

WebAgent provides two interchangeable workflows for LLM-judge evaluation of prediction files.

### Option 1: Evaluate during inference

Invoke `src/main.py` with the new evaluation flags. The CLI first writes predictions to `--output-path`, then immediately runs the judge and generates both a scored JSONL file and a summary report.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -m webagent.src.main \
  --agent webwalker \
  --dataset-name callanwu/WebWalkerQA \
  --dataset-split main \
  --output-path /tmp/webwalker_predictions.jsonl \
  --max-rounds 15 \
  --run-eval \
  --eval-output-path /tmp/webwalker_predictions_scored.jsonl \
  --judge-dataset webwalker \
  --force-rejudge
```

Flags of note:

- `--run-eval` enables evaluation once inference completes.
- `--eval-output-path` (optional) controls where the scored JSONL is written. Default: append `_eval` to `--output-path`.
- `--judge-dataset`, `--judge-model`, `--judge-prompt`, and `--force-rejudge` mirror the evaluator's parameters for dataset-specific prompts, model overrides, and re-scoring behaviour.

The helper script `run_main.sh` now uses this integrated path by default—adjust the environment variables at the top of the script and run `bash run_main.sh`.

### Option 2: Evaluate an existing predictions file

You can still call the standalone evaluator on any JSONL produced earlier (e.g., by the unified CLI or from archived runs):

```bash
python -m webagent.src.evaluate.evl \
  --input_path /tmp/webwalker_predictions.jsonl \
  --output_path /tmp/webwalker_predictions_scored.jsonl \
  --judge_dataset webwalker \
  --force-rejudge
```

Both workflows create:

- A scored JSONL (`*_scored.jsonl` / `*_eval.jsonl`) containing per-question scores and judge metadata.
- A companion summary report (`*_report.json`) with aggregate accuracy broken down by difficulty and source type.

Make sure the relevant LLM judge environment variables (OpenAI-compatible endpoint, keys, etc.) are configured before running either evaluation flow.

## Creating a custom agent

To add your own agent on top of this framework:

1. **Define tools (optional)**  
   - Implement new `BaseTool` subclasses in `src/tools/`, or reuse existing ones.  
   - Register them in `src/tools/__init__.py`, e.g. `MYAGENT_TOOLS`.

2. **Prepare prompts**  
   - Add a prompt module in `src/prompts/` that builds system/user prompts for the new agent.

3. **Implement the agent**  
   - Create `src/agents/myagent.py`, inherit `BaseAgent`, and implement `handle_user_message`, `generate_step_response`, `decide_next_action`, `process_tool_result`, `finalize_response`.  
   - Export the request dataclass and agent via `src/agents/__init__.py`.

4. **Register with the factory**  
   - Update `src/common/builder.py:create_agent` to route a new `--agent` value and return `AgentRunContext(agent=my_agent, request=...)`.

5. **Invoke from CLI**  
   - Run `python -m webagent.src.cli --agent myagent ...`. The unified CLI now streams the new agent’s reasoning/actions without further plumbing.

Both CLIs stream each workflow stage—prompts, model thoughts, tool invocations, tool responses, and final answers—directly to stdout. Interrupt safely with `Ctrl+C`.
