from web.search import search_web
from web.scraper import scrape_url
from web.cleaner import clean_html


# -----------------------------------------------------
# Web Ingestion Pipeline
# -----------------------------------------------------

def web_pipeline(
    query: str,
    max_results: int = 3,
) -> dict:
    """
    Complete web ingestion pipeline.

    Flow:
        Search -> Scrape -> Clean

    Args:
        query (str): User search query.
        max_results (int): Number of search results.

    Returns:
        {
            "success": bool,
            "documents": list,
            "error": str | None
        }

    Document Format:
        {
            "title": "...",
            "url": "...",
            "text": "..."
        }
    """

    try:

        # --------------------------------------------
        # Search
        # --------------------------------------------

        search_response = search_web(
            query=query,
            max_results=max_results,
        )

        if not search_response["success"]:
            return {
                "success": False,
                "documents": [],
                "error": search_response["error"],
            }

        search_results = search_response["results"]

        if not search_results:
            return {
                "success": False,
                "documents": [],
                "error": "No search results found.",
            }

        # --------------------------------------------
        # Process Search Results
        # --------------------------------------------

        documents = []

        for result in search_results:

            title = result["title"]
            url = result["url"]

            print(f"\nProcessing: {title}")

            # ----------------------------------------
            # Scrape
            # ----------------------------------------

            scrape_response = scrape_url(url)

            if not scrape_response["success"]:
                print(f"Failed to scrape: {url}")
                continue

            # ----------------------------------------
            # Clean HTML
            # ----------------------------------------

            clean_response = clean_html(
                scrape_response["html"]
            )

            if not clean_response["success"]:
                print(f"Failed to clean: {url}")
                continue

            # ----------------------------------------
            # Store Document
            # ----------------------------------------

            documents.append(
                {
                    "title": title,
                    "url": url,
                    "text": clean_response["text"],
                }
            )

            print("Success")

        # --------------------------------------------
        # Final Validation
        # --------------------------------------------

        if not documents:
            return {
                "success": False,
                "documents": [],
                "error": "No documents could be processed.",
            }

        return {
            "success": True,
            "documents": documents,
            "error": None,
        }

    except Exception as e:

        return {
            "success": False,
            "documents": [],
            "error": str(e),
        }


# -----------------------------------------------------
# Testing
# -----------------------------------------------------

if __name__ == "__main__":

    query = input("Enter your question: ")

    response = web_pipeline(query)

    if not response["success"]:
        print(f"\nError: {response['error']}")
        exit()

    print("\n")
    print("=" * 100)
    print(f"Documents Collected: {len(response['documents'])}")
    print("=" * 100)

    for index, document in enumerate(response["documents"], start=1):

        print(f"\nDocument {index}")
        print("-" * 100)

        print(f"Title : {document['title']}")
        print(f"URL   : {document['url']}")

        print("\nPreview:\n")
        print(document["text"][:1000])

        print("\n" + "=" * 100)