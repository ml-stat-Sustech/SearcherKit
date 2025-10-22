import json
import os
import time
from typing import Any, Dict, Optional

from tqdm import tqdm

from .utils import create_llm_judge_evaluator, load_webwalker_ground_truth


def eval_result(
    input_path: str,
    output_path: str,
    *,
    dataset: str = "webwalker",
    judge_model: Optional[str] = None,
    judge_prompt: Optional[str] = None,
    skip_existing: bool = True,
) -> None:
    """
    Evaluate prediction results against reference answers and generate a report.

    Parameters:
        input_path: Path to the input predictions file.
        output_path: Path to save the evaluation results and report.
        dataset: Dataset name used for llm_judge configuration.
        judge_model: Override model for llm_judge.
        judge_prompt: Override prompt template for llm_judge.
        skip_existing: When True, reuse scores already stored in output_path and
            only evaluate unseen questions. When False, re-evaluate all entries
            regardless of existing records.
    """
    info_lookup = load_webwalker_ground_truth()
    evaluator_fn, _ = create_llm_judge_evaluator(
        dataset=dataset,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
    )

    data_list = []
    visited = set()

    if not os.path.exists(output_path) or not skip_existing:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")

    if skip_existing and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                visited.add(json.loads(line)["question"])

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if skip_existing and data["question"] in visited:
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
        for data in data_list:
            try:
                outputs = call(data)
                print(outputs)
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
    parser.add_argument("--output_path", type=str, default='/mnt/sharedata/hdd/beier/Agent/WebWalker/llm_judge_eval.jsonl', help="Evaluation output path")
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
    parser.add_argument(
        "--force-rejudge",
        action="store_true",
        dest="force_rejudge",
        help="Re-evaluate all entries even if they already exist in the output file.",
    )
    args = parser.parse_args()

    eval_result(
        args.input_path,
        args.output_path,
        dataset=args.judge_dataset,
        judge_model=args.judge_model,
        judge_prompt=args.judge_prompt,
        skip_existing=not args.force_rejudge,
    )
