import json
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ..datasets import load_webwalker_ground_truth
from ..llm import LLMClient, OpenAIChatClient, build_llm_from_env


extracted_answer_format_for_confidence = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

extracted_answer_format_for_xbench = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "最终答案": {"type": "string"},
                "解释": {"type": "string"},
                "结论": {"type": "string", "enum": ["正确", "错误"]},
            },
            "required": ["最终答案", "解释", "结论"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

JUDGE_PROMPT_GAIA = """You are an impartial expert who decides whether a candidate answer matches the reference.

Question:
{question}

Reference Answer:
{correct_answer}

Candidate Answer:
{response}

Respond with either "Correct" if the candidate fully solves the task, or "Incorrect" otherwise. Please give response 
with Correct/Incorrect first, then you may optionally add a brief justification after the verdict.
"""

JUDGE_PROMPT_BROWSECOMP_OFFICIAL = """You are a grading assistant.

Question:
{question}

Reference Answer:
{correct_answer}

Candidate Answer:
{response}

Return a JSON object with the keys "extracted_final_answer", "reasoning", "correct" ("yes" or "no"), "confidence" (0-1), and "strict" (boolean). Ensure the JSON strictly matches the provided schema."""

JUDGE_PROMPT_XBENCH = """你是一个评测助手，需要判断候选答案是否与参考答案一致，并给出结论。

问题：
{question}

参考答案：
{correct_answer}

候选答案：
{response}

请返回一个 JSON，字段包括：
- "最终答案"：你判定的最终答案；
- "解释"：简要解释你的判断理由；
- "结论"：只能为"正确"或"错误"。
务必严格遵守 JSON 结构。"""

DEFAULT_JUDGE_CONFIGS: Dict[str, Tuple[str, str]] = {
    "gaia": (None, JUDGE_PROMPT_GAIA),
    "webwalker": (None, JUDGE_PROMPT_GAIA),
    "xbench-deepsearch": (None, JUDGE_PROMPT_XBENCH),
    "browsecomp_en": (None, JUDGE_PROMPT_BROWSECOMP_OFFICIAL),
    "browsecomp_en_full": (None, JUDGE_PROMPT_BROWSECOMP_OFFICIAL),
    "browsecomp_zh": (None, JUDGE_PROMPT_BROWSECOMP_OFFICIAL),
}

judge_prompt: Optional[str] = None
dataset: str = ""
judge_client: Optional[LLMClient] = None


def _build_judge_default_kwargs() -> Optional[Dict[str, object]]:
    """Collect default kwargs for judge completions from the environment."""

    default_kwargs: Dict[str, object] = {}
    temperature = os.environ.get("OPENAI_JUDGE_TEMPERATURE") or os.environ.get("OPENAI_TEMPERATURE")
    if temperature:
        default_kwargs["temperature"] = float(temperature)
    max_tokens = os.environ.get("OPENAI_JUDGE_MAX_OUTPUT_TOKENS") or os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
    if max_tokens:
        default_kwargs["max_tokens"] = int(max_tokens)
    return default_kwargs or None


def is_correct_judgement(judgement: str) -> bool:
    normalised = (judgement or "").strip().lower()
    if not normalised:
        return False

    if normalised.startswith("correct"):
        return True

    positive_prefixes = (
        "answer: correct",
        "answer : correct",
        "answer correct",
        "answer is correct",
        "verdict: correct",
        "verdict correct",
        "verdict is correct",
    )
    if any(normalised.startswith(prefix) for prefix in positive_prefixes):
        return True

    tokens = normalised.replace(":", " ").replace(".", " ").split()
    if tokens and tokens[0] == "correct":
        return True
    if len(tokens) >= 2 and tokens[0] in {"answer", "verdict"} and tokens[1] == "correct":
        return True
    if len(tokens) >= 3 and tokens[0] in {"answer", "verdict"} and tokens[1] == "is" and tokens[2] == "correct":
        return True

    return False


def configure_llm_judge(
    dataset_name: str,
    *,
    model_override: Optional[str] = None,
    prompt_override: Optional[str] = None,
    use_separate_judge_env: bool = False,
) -> None:
    global dataset, judge_prompt, judge_client

    dataset = dataset_name
    base_dataset = dataset_name
    if dataset_name.startswith("browsecomp_en"):
        base_dataset = "browsecomp_en"
    elif dataset_name.startswith("browsecomp_zh"):
        base_dataset = "browsecomp_zh"

    _, default_prompt = DEFAULT_JUDGE_CONFIGS.get(base_dataset, DEFAULT_JUDGE_CONFIGS["webwalker"])
    judge_prompt = prompt_override or default_prompt
    if judge_prompt is None:
        raise ValueError("judge_prompt is required for llm_judge evaluations.")

    if use_separate_judge_env:
        model_override = model_override or os.environ.get("OPENAI_JUDGE_MODEL")
        base_url = os.environ.get("OPENAI_JUDGE_MODEL_SERVER")
        api_key = os.environ.get("OPENAI_JUDGE_API_KEY")
        if not model_override:
            raise ValueError(
                "OPENAI_JUDGE_MODEL must be set when --use-separate-judge-llm is enabled for evaluation."
            )
        judge_client = OpenAIChatClient(
            model=model_override,
            base_url=base_url,
            api_key=api_key,
            default_kwargs=_build_judge_default_kwargs(),
        )
    elif model_override:
        judge_client = OpenAIChatClient(model=model_override)
    else:
        judge_client = build_llm_from_env()
    print(f"[llm judge] Using model: {judge_client.name()}")


def call_llm_judge(item: Dict[str, Any]) -> Dict[str, Any]:
    global judge_prompt, dataset, judge_client

    if judge_client is None or judge_prompt is None:
        raise ValueError("Judge client and prompt must be configured before calling the llm judge.")

    question = item["question"]
    correct_answer = item["answer"]
    response = (item.get("prediction") or item.get("pred") or "").strip()
    prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)

    for attempt in range(100):
        try:
            raw_details: Optional[Dict[str, Any]] = None
            raw_reply = judge_client.complete(
                [{"role": "user", "content": prompt}]
            )

            if dataset.startswith("xbench"):
                try:
                    parsed = json.loads(raw_reply)
                except json.JSONDecodeError:
                    parsed = {}
                raw_details = parsed if isinstance(parsed, dict) else {"raw": raw_reply}
                conclusion = (parsed or {}).get("结论", "")
                judgement = "Correct" if isinstance(conclusion, str) and conclusion.strip() == "正确" else ""
            elif "browsecomp" in dataset:
                try:
                    parsed = json.loads(raw_reply)
                except json.JSONDecodeError:
                    parsed = {}
                raw_details = parsed if isinstance(parsed, dict) else {"raw": raw_reply}
                correct_flag = (parsed or {}).get("correct", "")
                judgement = "Correct" if isinstance(correct_flag, str) and correct_flag.lower() == "yes" else ""
            else:
                judgement = raw_reply
                raw_details = {"response": raw_reply}

            return {
                "question": question,
                "answer": correct_answer,
                'prediction': response,
                "judgement": judgement,
                "details": raw_details,
            }
        except Exception as e:
            if attempt == 4:
                return {
                    "question": question,
                    "answer": correct_answer,
                    'prediction': response,
                    "judgement": "Error",
                    "error": str(e),
                }
            time.sleep(3)
            continue


def create_llm_judge_evaluator(
    *,
    dataset: str = "webwalker",
    judge_model: Optional[str] = None,
    judge_prompt: Optional[str] = None,
    use_separate_judge_env: bool = False,
) -> Tuple[Callable[[Dict[str, Any]], Dict[str, Any]], Optional[float]]:
    configure_llm_judge(
        dataset,
        model_override=judge_model,
        prompt_override=judge_prompt,
        use_separate_judge_env=use_separate_judge_env,
    )

    def _llm_judge_eval(data: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "question": data["question"],
            "answer": data["answer"],
            "prediction": data["pred"],
        }
        judge_response = call_llm_judge(item)
        is_correct = is_correct_judgement(judge_response.get("judgement", ""))
        return {
            "score": 1.0 if is_correct else 0.0,
            "raw": judge_response,
        }

    return _llm_judge_eval, None


def run_llm_judge_evaluation(
    *,
    input_path: str,
    output_path: str,
    dataset: str = "webwalker",
    judge_prompt: Optional[str] = None,
    skip_existing: bool,
    use_separate_judge_llm: bool,
    emit_func: Optional[Callable[..., None]] = None,
) -> None:
    """Execute the LLM judge evaluation workflow and emit progress logs."""

    def _emit(message: str, *, end: str = "\n") -> None:
        if emit_func is not None:
            emit_func(message, end=end)
        else:
            print(message, end=end)

    _emit("=" * 80)
    _emit("🧪 Running evaluation with LLM judge...")
    try:
        from .evl import eval_result  # Local import to avoid circular dependency

        eval_result(
            input_path,
            output_path,
            dataset=dataset,
            judge_prompt=judge_prompt,
            skip_existing=skip_existing,
            use_separate_judge_llm=use_separate_judge_llm,
        )
        abs_eval_output = os.path.abspath(output_path)
        _emit(f"📈 Evaluation scores saved to: {abs_eval_output}")
        report_base, _ = os.path.splitext(output_path)
        report_path = f"{report_base}_report.json"
        if os.path.exists(report_path):
            abs_report = os.path.abspath(report_path)
            _emit(f"🧾 Evaluation summary saved to: {abs_report}")
            try:
                with open(report_path, "r", encoding="utf-8") as report_handle:
                    report_data = json.load(report_handle)
                _emit(f"🔢 Overall accuracy: {report_data.get('overall')}")
            except Exception as report_exc:  # noqa: BLE001
                _emit(f"⚠️  Unable to read evaluation summary: {report_exc}")
    except Exception as eval_exc:  # noqa: BLE001
        _emit(f"❗ Evaluation failed: {eval_exc}")


__all__ = [
    "DEFAULT_JUDGE_CONFIGS",
    "call_llm_judge",
    "configure_llm_judge",
    "create_llm_judge_evaluator",
    "is_correct_judgement",
    "run_llm_judge_evaluation",
]
