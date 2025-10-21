import json
import os
import threading
import time
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import litellm
from datasets import load_dataset
from langchain.evaluation import load_evaluator
from openai import OpenAI


thread_local = threading.local()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")


def _get_openai_client() -> OpenAI:
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    return thread_local.client


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

Respond with either "Correct" if the candidate fully solves the task, or "Incorrect" otherwise. You may optionally add a brief justification after the verdict."""

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
    "gaia": ("openai/qwen2.5-72b-instruct", JUDGE_PROMPT_GAIA),
    "webwalker": ("openai/qwen2.5-72b-instruct", JUDGE_PROMPT_GAIA),
    "xbench-deepsearch": ("google/gemini-2.0-flash-001", JUDGE_PROMPT_XBENCH),
    "browsecomp_en": ("gpt-4o-2024-08-06", JUDGE_PROMPT_BROWSECOMP_OFFICIAL),
    "browsecomp_en_full": ("gpt-4o-2024-08-06", JUDGE_PROMPT_BROWSECOMP_OFFICIAL),
    "browsecomp_zh": ("gpt-4o-2024-08-06", JUDGE_PROMPT_BROWSECOMP_OFFICIAL),
}

judge_prompt: Optional[str] = None
judge_model: Optional[str] = None
dataset: str = ""


def is_correct_judgement(judgement: str) -> bool:
    judgement = (judgement or "").strip()
    if not judgement:
        return False
    lower = judgement.lower()
    return lower == "correct" or lower.startswith("answer: correct") or lower.startswith("answer:correct") or lower[0] == "a"


def configure_llm_judge(
    dataset_name: str,
    *,
    model_override: Optional[str] = None,
    prompt_override: Optional[str] = None,
) -> None:
    global dataset, judge_model, judge_prompt

    dataset = dataset_name
    base_dataset = dataset_name
    if dataset_name.startswith("browsecomp_en"):
        base_dataset = "browsecomp_en"
    elif dataset_name.startswith("browsecomp_zh"):
        base_dataset = "browsecomp_zh"

    default_model, default_prompt = DEFAULT_JUDGE_CONFIGS.get(
        base_dataset, DEFAULT_JUDGE_CONFIGS["webwalker"]
    )
    judge_model = model_override or default_model
    judge_prompt = prompt_override or default_prompt

    if judge_prompt is None:
        raise ValueError("judge_prompt is required for llm_judge evaluations.")


def call_llm_judge(item: Dict[str, Any]) -> Dict[str, Any]:
    global judge_prompt, dataset, judge_model

    if not judge_model or not judge_prompt:
        raise ValueError("Judge model and prompt must be configured before calling the llm judge.")

    question = item["question"]
    correct_answer = item["answer"]
    response = (item.get("prediction") or item.get("pred") or "").strip()
    prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)

    for attempt in range(100):
        try:
            raw_details: Optional[Dict[str, Any]] = None
            if judge_model == "openai/qwen2.5-72b-instruct":
                result = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5,
                )
                judgement = result.choices[0].message["content"]
                raw_details = {"response": judgement}
            elif judge_model == "google/gemini-2.0-flash-001":
                client = _get_openai_client()
                response_obj = client.beta.chat.completions.parse(
                    model=judge_model,
                    max_completion_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=extracted_answer_format_for_xbench,
                    timeout=100.0,
                )
                raw_judge = json.loads(response_obj.choices[0].message.content)
                judgement = "Correct" if raw_judge["结论"].lower() == "正确" else ""
                raw_details = raw_judge
            elif "browsecomp" in dataset:
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
                result = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5,
                    response_format=extracted_answer_format_for_confidence,
                )
                raw_content = result.choices[0].message["content"]
                raw_judge = json.loads(raw_content)
                judgement = "Correct" if raw_judge["correct"].lower() == "yes" else ""
                raw_details = raw_judge
            else:
                result = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5,
                )
                judgement = result.choices[0].message["content"]
                raw_details = {"response": judgement}

            return {
                "question": question,
                "answer": correct_answer,
                "judgement": judgement,
                "details": raw_details,
            }
        except Exception as e:
            if attempt == 4:
                return {
                    "question": question,
                    "answer": correct_answer,
                    "judgement": "Error",
                    "error": str(e),
                }
            time.sleep(3)
            continue


def create_evaluator(
    evaluator_type: str,
    *,
    dataset: str = "webwalker",
    judge_model: Optional[str] = None,
    judge_prompt: Optional[str] = None,
) -> Tuple[Callable[[Dict[str, Any]], Dict[str, Any]], Optional[float]]:
    evaluator_type = evaluator_type.lower()

    if evaluator_type == "langchain":
        evaluator = load_evaluator("cot_qa")

        def _langchain_eval(data: Dict[str, Any]) -> Dict[str, Any]:
            outputs = evaluator.evaluate_strings(
                prediction=data["pred"],
                input=data["question"],
                reference=data["answer"]
            )
            return {
                "score": outputs.get("score"),
                "raw": outputs,
            }

        return _langchain_eval, 4.0

    if evaluator_type == "llm_judge":
        configure_llm_judge(
            dataset,
            model_override=judge_model,
            prompt_override=judge_prompt,
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

    raise ValueError(f"Unsupported evaluator_type '{evaluator_type}'. Expected 'langchain' or 'llm_judge'.")


@lru_cache(maxsize=1)
def load_webwalker_ground_truth() -> Dict[str, Dict[str, Any]]:
    ds = load_dataset("callanwu/WebWalkerQA", split="main")
    info_adic: Dict[str, Dict[str, Any]] = {}
    for question, answer, info in zip(ds["question"], ds["answer"], ds["info"]):
        info_adic[question] = {
            "answer": answer,
            "info": info,
        }
    return info_adic


__all__ = [
    "DEFAULT_JUDGE_CONFIGS",
    "call_llm_judge",
    "configure_llm_judge",
    "create_evaluator",
    "is_correct_judgement",
    "load_webwalker_ground_truth",
]
