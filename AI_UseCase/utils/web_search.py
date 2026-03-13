"""
Web Search & Scrapy-style URL Scraping
Multi-tier: Tavily (premium) → DuckDuckGo search + full page scraping (free)

The key difference from basic search: we don't just return snippets —
we SCRAPE the full content of each result page using BeautifulSoup,
giving the LLM much richer context to answer from.
"""

import logging
import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import (
    WEB_SEARCH_PROMPT_TEMPLATE, BASE_SYSTEM_PROMPT,
    CONCISE_PROMPT, DETAILED_PROMPT,
    TAVILY_MAX_RESULTS, DDG_MAX_RESULTS,
)

logger = logging.getLogger(__name__)

# Shared headers for all HTTP requests (browser-like for best scraping results)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# =============================================================================
# SEARCH + SCRAPE (Search engine → full page scraping)
# =============================================================================

def search_and_scrape(query, max_results=DDG_MAX_RESULTS):
    """
    Search the web and scrape full content from result pages.

    Step 1: Search using DuckDuckGo → fallback to Bing scraping
    Step 2: Scrape each result page for full text (Scrapy-style extraction)
    Step 3: Return results with rich, full-page content (not just snippets)

    Args:
        query: Search query
        max_results: Number of results

    Returns:
        List of dicts with title, url, snippet (with FULL scraped content)
    """
    # Step 1: Get search result URLs (try DDG first, fallback to Bing)
    search_results = _ddg_search(query, max_results)
    if not search_results:
        logger.info("DuckDuckGo returned 0 results, falling back to Bing scraping")
        search_results = _bing_scrape_search(query, max_results)

    if not search_results:
        logger.warning(f"All search engines returned 0 results for: {query}")
        return []

    # Step 2: Scrape full content from each result page
    enriched = []
    for result in search_results[:max_results]:
        full_content = _scrape_page_content(result["url"])
        if full_content:
            result["snippet"] = full_content[:800]
            logger.info(f"  Scraped {len(full_content)} chars from: {result['url'][:60]}")
        enriched.append(result)

    logger.info(f"Search+Scrape: {len(enriched)} enriched results for: {query}")
    return enriched


def _ddg_search(query, max_results):
    """Search using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS(timeout=10) as ddgs:
            for r in ddgs.text(query, max_results=max_results + 2):
                url = r.get("href", "")
                if any(skip in url.lower() for skip in [
                    "youtube.com/watch", "maps.google", "accounts.google",
                ]):
                    continue
                results.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("body", "")[:300],
                })

        logger.info(f"DuckDuckGo: {len(results)} results for: {query}")
        return results

    except Exception as e:
        logger.error(f"DuckDuckGo error: {e}")
        return []


def _bing_scrape_search(query, max_results):
    """
    Scrape Bing search results as fallback when DuckDuckGo is unavailable.
    Uses requests + BeautifulSoup (Scrapy-style scraping).
    """
    try:
        import urllib.parse

        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded_query}&count={max_results + 5}"

        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for li in soup.select("li.b_algo"):
            h2 = li.find("h2")
            a_tag = h2.find("a") if h2 else None
            if not a_tag:
                continue

            title = a_tag.get_text(strip=True)
            href = a_tag.get("href", "")

            # Bing sometimes uses redirect URLs — try to follow them
            if "bing.com/ck/" in href:
                try:
                    head_resp = requests.head(href, headers=_HEADERS, timeout=5, allow_redirects=True)
                    href = head_resp.url
                except Exception:
                    continue  # Skip if can't resolve

            # Skip non-http URLs
            if not href.startswith("http"):
                continue

            # Get snippet text
            snippet_el = li.find("p") or li.find("div", class_="b_caption")
            snippet = snippet_el.get_text(strip=True)[:300] if snippet_el else ""

            results.append({
                "title": title,
                "url": href,
                "snippet": snippet,
            })

            if len(results) >= max_results:
                break

        logger.info(f"Bing scrape: {len(results)} results for: {query}")
        return results

    except Exception as e:
        logger.error(f"Bing scrape error: {e}")
        return []


def _scrape_page_content(url):
    """
    Scrape full text content from a page (Scrapy-style extraction).
    Returns clean text string or None on failure.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "noscript", "iframe", "form", "button", "svg", "img"]):
            tag.decompose()

        main = (
            soup.find("main") or soup.find("article") or
            soup.find("div", {"role": "main"}) or
            soup.find("div", {"id": "content"}) or
            soup.find("div", {"class": "content"}) or
            soup
        )

        text = main.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 10]
        clean = "\n".join(lines)

        return clean if len(clean) >= 50 else None

    except Exception:
        return None


# =============================================================================
# TAVILY SEARCH (premium, LLM-optimized)
# =============================================================================

def tavily_search(query, api_key, max_results=TAVILY_MAX_RESULTS):
    """Search using Tavily API (premium, returns LLM-optimized content)."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:500],
            })

        logger.info(f"Tavily returned {len(results)} results for: {query}")
        return results

    except Exception as e:
        logger.error(f"Tavily error: {e}")
        return []


# =============================================================================
# UNIFIED WEB SEARCH
# =============================================================================

def web_search(query, tavily_key=None):
    """
    Perform web search with automatic fallback + full page scraping.

    Tier 1: Tavily (if API key) — premium, LLM-optimized content
    Tier 2: DuckDuckGo search + Scrapy-style page scraping (free)

    Returns:
        List of dicts with title, url, snippet (rich scraped content)
    """
    if tavily_key:
        results = tavily_search(query, tavily_key)
        if results:
            return results

    return search_and_scrape(query)


# =============================================================================
# URL SCRAPING (for sidebar "Add URL" feature)
# =============================================================================

def scrape_url(url):
    """
    Scrape text content from a URL. Handles HTML pages and PDF documents.

    Returns:
        dict with keys: text, title, char_count, content_type
        Returns None on failure.
    """
    try:
        response = requests.get(url, headers=_HEADERS, timeout=15, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        try:
            logger.warning(f"SSL error for {url}, retrying without verification")
            response = requests.get(url, headers=_HEADERS, timeout=15, verify=False, allow_redirects=True)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"URL fetch failed (SSL retry): {url} — {e}")
            return None
    except Exception as e:
        logger.error(f"URL fetch failed: {url} — {e}")
        return None

    content_type = response.headers.get("Content-Type", "").lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return _scrape_pdf_url(response.content, url)

    return _scrape_html(response.text, url)


def _scrape_pdf_url(pdf_bytes, url):
    """Extract text from a PDF downloaded from a URL."""
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
            logger.warning(f"PDF at {url} has no extractable text")
            return None

        full_text = "\n\n".join(pages_text)
        title = url.split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ").title()

        logger.info(f"Extracted {len(full_text)} chars from PDF ({len(reader.pages)} pages): {url}")
        return {
            "text": full_text[:50000],
            "title": title,
            "char_count": len(full_text),
            "content_type": "pdf",
            "page_count": len(reader.pages),
        }
    except Exception as e:
        logger.error(f"PDF extraction failed for {url}: {e}")
        return None


def _scrape_html(html_content, url):
    """Extract clean text from HTML content."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "noscript", "iframe", "form", "button", "svg", "img"]):
            tag.decompose()

        for tag in soup.find_all(style=True):
            style = (tag.get("style") or "").replace(" ", "")
            if "display:none" in style or "visibility:hidden" in style:
                tag.decompose()

        main_content = (
            soup.find("main") or soup.find("article") or
            soup.find("div", {"role": "main"}) or
            soup.find("div", {"id": "content"}) or
            soup.find("div", {"class": "content"}) or
            soup
        )

        text = main_content.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 2]
        clean_text = "\n".join(lines)

        if len(clean_text) < 50:
            return None

        logger.info(f"Scraped {len(clean_text)} chars from HTML: {url}")
        return {
            "text": clean_text[:50000],
            "title": title,
            "char_count": len(clean_text),
            "content_type": "html",
        }
    except Exception as e:
        logger.error(f"HTML extraction failed for {url}: {e}")
        return None


# =============================================================================
# WEB AUGMENTED QUERY (legacy functions)
# =============================================================================

def web_augmented_query(query, llm, response_mode="concise", chat_history=None, tavily_key=None):
    """Perform web search and synthesize results with LLM."""
    try:
        search_results = web_search(query, tavily_key)
        if not search_results:
            return {
                "answer": "I couldn't find relevant web search results. Please try rephrasing your question.",
                "sources": [], "reasoning": "Web search returned no results.",
            }

        formatted_results = _format_results(search_results)
        web_prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(search_results=formatted_results, question=query)
        mode_prompt = CONCISE_PROMPT if response_mode == "concise" else DETAILED_PROMPT
        messages = [SystemMessage(content=f"{BASE_SYSTEM_PROMPT}\n\n{mode_prompt}")]
        if chat_history:
            from langchain_core.messages import HumanMessage as HM, AIMessage as AM
            for msg in (chat_history or [])[-6:]:
                cls = HM if msg["role"] == "user" else AM
                messages.append(cls(content=msg["content"]))
        messages.append(HumanMessage(content=web_prompt))
        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": search_results,
            "reasoning": f"Searched web and found {len(search_results)} result(s).",
        }
    except Exception as e:
        logger.error(f"Web augmented query error: {e}")
        return {"answer": f"An error occurred: {str(e)}", "sources": [], "reasoning": f"Error: {str(e)}"}


def web_augmented_query_stream(query, llm, response_mode="concise", chat_history=None, tavily_key=None):
    """Streaming version of web_augmented_query."""
    try:
        search_results = web_search(query, tavily_key)
        if not search_results:
            yield {"type": "complete", "answer": "No web results found.", "sources": [], "reasoning": "No results."}
            return

        formatted_results = _format_results(search_results)
        web_prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(search_results=formatted_results, question=query)
        mode_prompt = CONCISE_PROMPT if response_mode == "concise" else DETAILED_PROMPT
        messages = [SystemMessage(content=f"{BASE_SYSTEM_PROMPT}\n\n{mode_prompt}")]
        if chat_history:
            from langchain_core.messages import HumanMessage as HM, AIMessage as AM
            for msg in (chat_history or [])[-6:]:
                cls = HM if msg["role"] == "user" else AM
                messages.append(cls(content=msg["content"]))
        messages.append(HumanMessage(content=web_prompt))

        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_response += token
                yield {"type": "token", "token": token}

        yield {
            "type": "complete", "answer": full_response,
            "sources": search_results, "reasoning": f"Searched web: {len(search_results)} results.",
        }
    except Exception as e:
        logger.error(f"Web stream error: {e}")
        yield {"type": "complete", "answer": f"Error: {str(e)}", "sources": [], "reasoning": f"Error: {str(e)}"}


def _format_results(search_results):
    """Format search results for LLM prompt."""
    formatted = ""
    for i, r in enumerate(search_results):
        formatted += f"\n[Source {i+1}] {r['title']}\nURL: {r['url']}\nContent: {r['snippet']}\n"
    return formatted
