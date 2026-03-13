"""
Website Loader — Crawl company website pages and extract policy content.
Supports single-page scraping and multi-page crawling (follows internal links).
"""

import logging
import re
from urllib.parse import urlparse, urljoin
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_website(url, max_pages=10):
    """
    Crawl a website starting from the given URL.
    Follows internal links up to max_pages.

    Args:
        url: Starting URL (e.g. https://company.com/policies)
        max_pages: Maximum number of pages to crawl

    Returns:
        List of LangChain Document objects (one per page)
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc
    visited = set()
    to_visit = [url]
    documents = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            resp = requests.get(current_url, headers=headers, timeout=15, allow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Failed to fetch {current_url}: {e}")
            visited.add(current_url)
            continue

        visited.add(current_url)
        content_type = resp.headers.get("Content-Type", "").lower()

        # Handle PDFs
        if "application/pdf" in content_type or current_url.lower().endswith(".pdf"):
            doc = _extract_pdf(resp.content, current_url)
            if doc:
                documents.append(doc)
            continue

        # Parse HTML
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract document
        doc = _extract_page(soup, current_url)
        if doc:
            documents.append(doc)

        # Find internal links to follow
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)

            # Only follow same-domain links, skip anchors/mailto/tel
            if (parsed.netloc == base_domain
                    and parsed.scheme in ("http", "https")
                    and full_url not in visited
                    and not parsed.fragment
                    and not parsed.path.endswith((".jpg", ".png", ".gif", ".css", ".js", ".zip"))):
                to_visit.append(full_url)

    logger.info(f"Website crawl complete: {len(documents)} pages from {url} (visited {len(visited)} URLs)")
    return documents


def load_single_page(url):
    """
    Scrape a single URL and return as a LangChain Document.

    Args:
        url: URL to scrape

    Returns:
        Document or None
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        try:
            resp = requests.get(url, headers=headers, timeout=15, verify=False, allow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

    content_type = resp.headers.get("Content-Type", "").lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return _extract_pdf(resp.content, url)

    soup = BeautifulSoup(resp.text, "html.parser")
    return _extract_page(soup, url)


def _extract_page(soup, url):
    """Extract clean text from a parsed HTML page."""
    # Get title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1] or "Untitled"

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "form", "button", "svg", "img"]):
        tag.decompose()

    # Remove hidden elements
    for tag in soup.find_all(style=True):
        style = (tag.get("style") or "").replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()

    # Prefer main content area
    main_content = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", {"role": "main"}) or
        soup.find("div", {"id": "content"}) or
        soup.find("div", {"class": "content"}) or
        soup
    )

    text = main_content.get_text(separator="\n", strip=True)

    # Clean up
    lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 2]
    clean_text = "\n".join(lines)

    if len(clean_text) < 50:
        logger.warning(f"Too little content from {url} ({len(clean_text)} chars)")
        return None

    return Document(
        page_content=clean_text[:50000],
        metadata={
            "source": title,
            "url": url,
            "page": 1,
            "type": "website",
            "char_count": len(clean_text),
        },
    )


def _extract_pdf(pdf_bytes, url):
    """Extract text from PDF bytes."""
    try:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text.strip())

        if not pages_text:
            return None

        full_text = "\n\n".join(pages_text)
        title = url.split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ").title()

        return Document(
            page_content=full_text[:50000],
            metadata={
                "source": title,
                "url": url,
                "page": len(reader.pages),
                "type": "pdf_url",
                "char_count": len(full_text),
            },
        )
    except Exception as e:
        logger.error(f"PDF extraction failed for {url}: {e}")
        return None
