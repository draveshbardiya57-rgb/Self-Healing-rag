import re

from bs4 import BeautifulSoup, Comment


# -----------------------------------------------------
# Config
# -----------------------------------------------------

MAX_TEXT_LENGTH = 5000  # cap output so downstream critic prompts stay bounded

UNWANTED_TAGS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "button",
    "svg",
    "canvas",
    "iframe",
]


# -----------------------------------------------------
# HTML Cleaning Function
# -----------------------------------------------------

def clean_html(html: str) -> dict:
    """
    Clean raw HTML and extract readable text.

    Preserves <pre> and <code> blocks as fenced/backtick text so
    technical content (code snippets, API examples) doesn't get
    flattened into unreadable runs of tokens by get_text().

    Args:
        html (str): Raw HTML downloaded from a webpage.

    Returns:
        {
            "success": bool,
            "text": str,
            "error": str | None
        }
    """

    try:
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # -------------------------------------------------
        # Try to extract only the main content
        # -------------------------------------------------

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", id="mw-content-text")      # Wikipedia
            or soup.find("div", {"role": "main"})
            or soup
        )

        # -------------------------------------------------
        # Preserve code blocks BEFORE removing/flattening
        # -------------------------------------------------
        # get_text() collapses whitespace inside these tags too,
        # so we replace them with fenced markers first.

        for pre_tag in main_content.find_all("pre"):
            code_text = pre_tag.get_text()
            pre_tag.replace_with(f"\n```\n{code_text}\n```\n")

        for code_tag in main_content.find_all("code"):
            code_text = code_tag.get_text()
            code_tag.replace_with(f"`{code_text}`")

        # -------------------------------------------------
        # Remove unwanted HTML elements
        # -------------------------------------------------

        for tag in main_content(UNWANTED_TAGS):
            tag.decompose()

        # -------------------------------------------------
        # Remove HTML comments
        # -------------------------------------------------

        for comment in main_content.find_all(
            string=lambda text: isinstance(text, Comment)
        ):
            comment.extract()

        # -------------------------------------------------
        # Extract visible text
        # -------------------------------------------------

        text = main_content.get_text(separator=" ", strip=True)

        # -------------------------------------------------
        # Normalize whitespace
        # -------------------------------------------------

        text = re.sub(r"\s+", " ", text)

        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        text = text.strip()

        # -------------------------------------------------
        # Cap length — protects downstream prompt budgets
        # -------------------------------------------------

        text = text[:MAX_TEXT_LENGTH]

        return {
            "success": True,
            "text": text,
            "error": None,
        }

    except Exception as e:

        return {
            "success": False,
            "text": "",
            "error": str(e),
        }


# -----------------------------------------------------
# Testing
# -----------------------------------------------------

if __name__ == "__main__":

    from web.scraper import scrape_url

    url = input("Enter URL: ").strip()

    scrape_response = scrape_url(url)

    if not scrape_response["success"]:
        print(scrape_response["error"])
        exit()

    clean_response = clean_html(scrape_response["html"])

    if not clean_response["success"]:
        print(clean_response["error"])
        exit()

    print("\nCleaned Text\n")
    print("=" * 100)
    print(clean_response["text"][:3000])
    print("=" * 100)
    print(f"\nTotal Characters: {len(clean_response['text'])}")