import os
import re
import urllib.parse
from typing import Tuple

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
