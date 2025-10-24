import json
import os
import re
import urllib.parse
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

from .state import STATE, WebWalkerState


def process_url(url: str, sub_url: str) -> str:
    """Return an absolute URL for a hyperlink found on the page."""
    return urllib.parse.urljoin(url, sub_url)


def clean_markdown(res: str) -> str:
    """Remove redundant hyperlinks from markdown text."""
    pattern = r"\[.*?\]\(.*?\)"
    try:
        result = re.sub(pattern, "", res)
        url_pattern = (
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]"
            r"|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        result = re.sub(url_pattern, "", result)
        result = result.replace("* \n", "")
        result = re.sub(r"\n\n+", "\n", result)
        return result
    except Exception:
        return res


async def get_info(url: str, screenshot: bool = False) -> Tuple[str, str, str]:
    """Fetch HTML, cleaned markdown, and optionally a screenshot for a URL."""
    run_config = CrawlerRunConfig(
        screenshot=screenshot,
        screenshot_wait_for=1.0,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=run_config)
    return result.html, clean_markdown(result.markdown), result.screenshot


def extract_links_with_text(html: str, state: WebWalkerState = STATE) -> str:
    """Extract in-domain anchor/button labels and update the CLI state."""
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a_tag in soup.find_all("a", href=True):
        url = a_tag["href"]
        text = "".join(a_tag.stripped_strings)
        if text and "javascript" not in url and not url.lower().endswith(
            (".jpg", ".png", ".gif", ".jpeg", ".pdf")
        ):
            full_url = process_url(state.root_url, url)
            if full_url.startswith(state.root_url):
                links.append({"url": full_url, "text": text})

    for a_tag in soup.find_all("a", onclick=True):
        onclick_text = a_tag["onclick"]
        text = "".join(a_tag.stripped_strings)
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if text and url and not url.lower().endswith(
                (".jpg", ".png", ".gif", ".jpeg", ".pdf")
            ):
                full_url = process_url(state.root_url, url)
                if full_url.startswith(state.root_url):
                    links.append({"url": full_url, "text": text})

    for a_tag in soup.find_all("a", attrs={"data-url": True}):
        url = a_tag["data-url"]
        text = "".join(a_tag.stripped_strings)
        if text and url and not url.lower().endswith(
            (".jpg", ".png", ".gif", ".jpeg", ".pdf")
        ):
            full_url = process_url(state.root_url, url)
            if full_url.startswith(state.root_url):
                links.append({"url": full_url, "text": text})

    for a_tag in soup.find_all("a", class_="herf-mask"):
        url = a_tag.get("href")
        text = a_tag.get("title") or "".join(a_tag.stripped_strings)
        if text and url and not url.lower().endswith(
            (".jpg", ".png", ".gif", ".jpeg", ".pdf")
        ):
            full_url = process_url(state.root_url, url)
            if full_url.startswith(state.root_url):
                links.append({"url": full_url, "text": text})

    for button in soup.find_all("button", onclick=True):
        onclick_text = button["onclick"]
        text = (
            button.get("title")
            or button.get("aria-label")
            or "".join(button.stripped_strings)
        )
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match and text:
            url = match.group(1)
            full_url = process_url(state.root_url, url)
            if full_url.startswith(state.root_url):
                links.append({"url": full_url, "text": text})

    unique_links = {
        f"{item['url']}_{item['text']}": item for item in links if item["text"]
    }

    info_lines = []
    for link in unique_links.values():
        label = link["text"]
        state.register_button(label, link["url"])
        info_lines.append(f"<button>{label}<button>")

    return "\n".join(info_lines)


def get_content_between_a_b(start_tag: str, end_tag: str, text: str) -> str:
    """Return all substrings enclosed by start_tag and end_tag."""
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break
    return extracted_text.strip()


# Dataset field selection helpers -------------------------------------------------

QUESTION_FIELD_CANDIDATES: Tuple[str, ...] = (
    "question",
    "Question",
    "query",
    "prompt",
    "instruction",
)
ANSWER_FIELD_CANDIDATES: Tuple[str, ...] = (
    "answer",
    "Answer",
    "final_answer",
    "Final answer",
    "Final Answer",
    "outputs",
    "response",
)
ROOT_URL_FIELD_CANDIDATES: Tuple[str, ...] = (
    "root_url",
    "root",
    "root_url_full",
    "Root URL",
    "website",
    "target_website",
)
INFO_FIELD_CANDIDATES: Tuple[str, ...] = (
    "info",
    "Info",
    "metadata",
    "Metadata",
    "Annotator Metadata",
)
EXTRA_METADATA_FIELDS: Tuple[str, ...] = (
    "task_id",
    "Task ID",
    "TaskID",
    "Level",
    "file_name",
    "file_path",
    "category",
)


def default_eval_output_path(prediction_path: str) -> str:
    """Return <path>_eval.jsonl when no custom evaluation output is provided."""
    base, ext = os.path.splitext(prediction_path)
    if ext:
        return f"{base}_eval{ext}"
    return f"{prediction_path}_eval.jsonl"


def first_non_empty(sample: Mapping[str, Any], candidates: Iterable[str]) -> Optional[Any]:
    """Return the first non-empty value in sample whose key appears in candidates."""
    for key in candidates:
        if key not in sample:
            continue
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            return stripped
        return value
    return None


def collect_metadata(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract a compact metadata dictionary for downstream logging."""
    metadata: Dict[str, Any] = {}
    for key in EXTRA_METADATA_FIELDS:
        if key not in sample:
            continue
        value = sample.get(key)
        if value in (None, "", []):
            continue
        metadata[key] = value
    return metadata


def extract_sample_fields(sample: Mapping[str, Any]) -> Tuple[Optional[str], Optional[Any], Optional[str], Optional[Any], Dict[str, Any]]:
    """Return key fields (question/answer/root/info/metadata) from a dataset sample."""
    question = first_non_empty(sample, QUESTION_FIELD_CANDIDATES)
    answer = first_non_empty(sample, ANSWER_FIELD_CANDIDATES)
    root_url = first_non_empty(sample, ROOT_URL_FIELD_CANDIDATES)

    info = None
    for key in INFO_FIELD_CANDIDATES:
        if key in sample:
            info = sample.get(key)
            if info not in (None, "", []):
                break

    metadata = collect_metadata(sample)
    return question, answer, root_url, info, metadata


def apply_local_wiki_env_flags(args) -> None:
    """Populate environment variables expected by local wiki tooling."""
    if getattr(args, "use_local_wiki_tools", False):
        os.environ["WEBDANCER_USE_LOCAL_WIKI"] = "1"

    env_mapping = {
        "local_wiki_index": "LOCAL_WIKI_INDEX",
        "local_wiki_es_host": "LOCAL_WIKI_ES_HOST",
        "local_wiki_es_timeout": "LOCAL_WIKI_ES_TIMEOUT",
        "local_wiki_retriever": "LOCAL_WIKI_RETRIEVER",
        "local_wiki_model_name": "LOCAL_WIKI_MODEL_NAME",
    }
    for arg_name, env_name in env_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            os.environ[env_name] = str(value)


def load_existing_questions(output_path: str) -> Tuple[set[str], Optional[str]]:
    """Return existing questions from a JSONL file and an optional warning."""
    existing_questions: set[str] = set()
    if not os.path.exists(output_path):
        return existing_questions, None

    try:
        with open(output_path, "r", encoding="utf-8") as existing_handle:
            for line in existing_handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question_value = entry.get("question")
                if isinstance(question_value, str):
                    existing_questions.add(question_value)
    except Exception as exc:  # noqa: BLE001
        return existing_questions, f"⚠️  Unable to read existing output file: {exc}"

    return existing_questions, None


def write_prediction_record(
    handle,
    *,
    question: str,
    prediction: str,
    answer: Optional[Any] = None,
    root_url: Optional[str] = None,
    info: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    fallback_task_id: Optional[Any] = None,
) -> Dict[str, Any]:
    """Serialise a prediction entry to JSONL and flush the target handle."""
    record: Dict[str, Any] = {
        "question": question,
        "pred": prediction,
    }
    if answer not in (None, "", []):
        record["answer"] = answer
    if root_url:
        record["root_url"] = root_url
    if info not in (None, "", []):
        record["info"] = info
    model_metadata = metadata or {}
    if not model_metadata and fallback_task_id:
        model_metadata = {"task_id": fallback_task_id}
    if model_metadata:
        record["metadata"] = model_metadata

    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    return record
