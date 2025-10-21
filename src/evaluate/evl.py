import json
import os
import time
import concurrent.futures
from typing import Any, Dict, Optional

from tqdm import tqdm

from .utils import create_evaluator, load_webwalker_ground_truth


def eval_result(
    input_path: str,
    output_path: str,
    *,
    evaluator_type: str = "langchain",
    dataset: str = "webwalker",
    judge_model: Optional[str] = None,
    judge_prompt: Optional[str] = None,
) -> None:
    """
    Evaluate prediction results against reference answers and generate a report.

    Parameters:
        input_path: Path to the input predictions file.
        output_path: Path to save the evaluation results and report.
        evaluator_type: Which evaluator to use ('langchain' or 'llm_judge').
        dataset: Dataset name used for llm_judge configuration.
        judge_model: Override model for llm_judge.
        judge_prompt: Override prompt template for llm_judge.
    """
    info_lookup = load_webwalker_ground_truth()
    evaluator_fn, future_timeout = create_evaluator(
        evaluator_type,
        dataset=dataset,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
    )

    data_list = []
    visited = []

    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            visited.append(json.loads(line)["question"])

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["question"] in visited:
                continue
            gt = info_lookup.get(data["question"])
            if gt:
                data["answer"] = gt["answer"]
                data_list.append(data)

    def call(data: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return evaluator_fn(data)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))
                else:
                    raise e

    s = 0
    cnt = 0

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_data = {executor.submit(call, data): data for data in data_list}
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    outputs = future.result(timeout=future_timeout) if future_timeout else future.result()
                    data = future_to_data[future]
                    data["score"] = outputs["score"]
                    if outputs.get("raw") is not None:
                        data["judge_details"] = outputs["raw"]

                    cnt += data["score"]
                    s += 1

                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

                    pbar.update(1)
                    print("Current accuracy:", cnt / s)

                except Exception as e:
                    print(f"Error processing data: {e}")

    single_source_easy, single_source_medium, single_source_hard = [], [], []
    multi_source_easy, multi_source_medium, multi_source_hard = [], [], []
    overall = []

    datas = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gt = info_lookup.get(item["question"])
            if gt:
                item["info"] = gt["info"]
                datas.append(item)

    for temp in datas:
        score = temp.get("score")
        if score is None:
            continue
        info = temp.get("info", {})
        q_type = info.get("type")
        difficulty = info.get("difficulty_level")

        if q_type == "single_source":
            if difficulty == "easy":
                single_source_easy.append(score)
            elif difficulty == "medium":
                single_source_medium.append(score)
            elif difficulty == "hard":
                single_source_hard.append(score)

        elif q_type == "multi_source":
            if difficulty == "easy":
                multi_source_easy.append(score)
            elif difficulty == "medium":
                multi_source_medium.append(score)
            elif difficulty == "hard":
                multi_source_hard.append(score)

        overall.append(score)

    def safe_average(scores):
        return sum(scores) / len(scores) if scores else None

    result = {
        "single_source_easy": safe_average(single_source_easy),
        "single_source_medium": safe_average(single_source_medium),
        "single_source_hard": safe_average(single_source_hard),
        "multi_source_easy": safe_average(multi_source_easy),
        "multi_source_medium": safe_average(multi_source_medium),
        "multi_source_hard": safe_average(multi_source_hard),
        "overall": safe_average(overall),
    }

    report_path = output_path.split(".jsonl")[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    abs_output = os.path.abspath(output_path)
    abs_report = os.path.abspath(report_path)
    print(f"[evl] Evaluation log saved to: {abs_output}")
    print(f"[evl] Summary report saved to: {abs_report}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Input prediction result path")
    parser.add_argument("--output_path", type=str, help="Evaluation output path")
    parser.add_argument(
        "--evaluator",
        type=str,
        default="langchain",
        choices=["langchain", "llm_judge"],
        help="Choose evaluator backend",
    )
    parser.add_argument(
        "--judge_dataset",
        type=str,
        default="webwalker",
        help="Dataset identifier for llm_judge configuration",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Override model name for llm_judge",
    )
    parser.add_argument(
        "--judge_prompt",
        type=str,
        default=None,
        help="Override prompt template for llm_judge",
    )
    args = parser.parse_args()

    eval_result(
        args.input_path,
        args.output_path,
        evaluator_type=args.evaluator,
        dataset=args.judge_dataset,
        judge_model=args.judge_model,
        judge_prompt=args.judge_prompt,
    )
