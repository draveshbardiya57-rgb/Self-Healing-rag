import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# =========================
# CONFIG
# =========================

START_URL = "https://fastapi.tiangolo.com/"
OUTPUT_DIR = "documents/developer_docs/Essential_Dev_Docs/fastapi"

MAX_PAGES = 400
DELAY = 1

# Full FastAPI translated-language prefix list — includes hyphenated
# regional variants like zh-hant, which the earlier version missed
LANG_PREFIXES = (
    "/az/", "/de/", "/em/", "/es/", "/fa/", "/fr/", "/he/", "/hi/",
    "/id/", "/it/", "/ja/", "/ko/", "/nl/", "/pl/", "/pt/", "/ru/",
    "/tr/", "/uk/", "/ur/", "/vi/", "/yo/", "/zh/", "/zh-hant/",
)

visited = set()
queue = [START_URL]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================

def clean_filename(url):
    path = urlparse(url).path

    if path == "/":
        path = "index"

    path = path.strip("/")

    filename = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        path
    )

    if not filename:
        filename = "index"

    return filename + ".txt"


def fix_encoding(text):
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "Â¶": "",
        "Â": "",
        "\xa0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

def strip_residual_html_tags(text):
    """
    Catches HTML tags that survive extraction as literal text
    rather than being stripped as real DOM elements. Seen in
    MkDocs terminal-output blocks that embed styled markup
    (font color, span, u tags) inside <pre> as escaped content.
    """
    # Remove any <tag ...> or </tag> pattern, keeping inner text
    text = re.sub(r"<[^>]+>", "", text)
    return text
def remove_noise(text):
    # FastAPI uses MkDocs + Material theme — different boilerplate
    # than Python's Sphinx docs
    noise_patterns = [
        r"\bSkip to content\b",
        r"\bBack to top\b",
        r"\bEdit this page\b",
        r"\bnext\b",
        r"\bprevious\b",
        r"\bTable of contents\b",
        r"\bWas this page helpful\?\b",
        r"\bThanks for your feedback!\b",
        r"^\s*\*\s*$",
    ]

    for pattern in noise_patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "nav",
        "footer"
    ]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("div", class_="md-content")
        or soup.find("main")
    )

    if not main:
        main = soup

    # -------------------------------------------------
    # CRITICAL FIX — preserve code blocks before get_text()
    # flattens them. Without this, <pre> blocks (code
    # examples) and inline syntax-highlighted terminal
    # output (<font>/<span> wrapped) get destroyed:
    # every token ends up on its own line, and highlighting
    # tags leak into the text as literal <font color=...>.
    #
    # pre_tag.get_text() strips the inner <font>/<span>
    # wrapper tags automatically, keeping only the real text.
    # -------------------------------------------------

    for pre_tag in main.find_all("pre"):
        code_text = pre_tag.get_text()
        pre_tag.replace_with(f"\n```\n{code_text}\n```\n")

    for code_tag in main.find_all("code"):
        code_text = code_tag.get_text()
        code_tag.replace_with(f"`{code_text}`")

    text = main.get_text(
        separator="\n",
        strip=True
    )

    text = fix_encoding(text)
    text = remove_noise(text)

    return text


# =========================
# MAIN CRAWLER
# =========================

domain = urlparse(START_URL).netloc

count = 0

while queue and count < MAX_PAGES:

    url = queue.pop(0)

    if url in visited:
        continue

    visited.add(url)

    try:

        print(f"[{count + 1}] Crawling: {url}")

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "DeveloperDocsCrawler/1.0"
            }
        )

        if response.status_code != 200:
            continue

        text = extract_text(response.text)

        if len(text) < 200:
            print(f"  Skipping — too short ({len(text)} chars)")
            count += 1
            continue

        filename = clean_filename(url)

        filepath = os.path.join(
            OUTPUT_DIR,
            filename
        )

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"SOURCE_URL: {url}\n\n"
            )

            f.write(text)

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for link in soup.find_all(
            "a",
            href=True
        ):

            next_url = urljoin(
                url,
                link["href"]
            )

            parsed = urlparse(next_url)

            # Stay on fastapi.tiangolo.com only
            if parsed.netloc != domain:
                continue

            # Skip translated-language pages
            if parsed.path.startswith(LANG_PREFIXES):
                continue

            # Remove fragments
            next_url = (
                parsed.scheme
                + "://"
                + parsed.netloc
                + parsed.path
            )

            if next_url not in visited and next_url not in queue:
                queue.append(next_url)

        count += 1

        time.sleep(DELAY)

    except Exception as e:
        print(f"ERROR: {url}")
        print(e)

print(f"\nDownloaded {count} pages")