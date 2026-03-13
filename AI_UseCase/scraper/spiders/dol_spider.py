"""
DOL (Department of Labor) Spider
Scrapes labor law topics, wage/hour guidance, and compliance info from dol.gov
"""

import scrapy
from datetime import datetime, timezone
from scraper.items import PolicyDocument


class DolSpider(scrapy.Spider):
    name = "dol"
    allowed_domains = ["dol.gov"]
    start_urls = [
        "https://www.dol.gov/general/topics",
        "https://www.dol.gov/agencies/whd",
        "https://www.dol.gov/general/topic/wages",
        "https://www.dol.gov/general/topic/benefits-leave",
        "https://www.dol.gov/general/topic/health-plans",
        "https://www.dol.gov/general/topic/retirement",
        "https://www.dol.gov/general/topic/disability",
        "https://www.dol.gov/general/topic/workhours",
        "https://www.dol.gov/general/topic/whistleblower",
        "https://www.dol.gov/general/topic/termination",
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2,
    }

    def parse(self, response):
        """Parse DOL topic index pages and follow sub-topic links."""
        # Extract content from current page
        yield from self._extract_page(response)

        # Follow sub-topic links
        links = response.css("a[href]::attr(href)").getall()
        seen = set()

        target_paths = [
            "/general/topic/",
            "/agencies/whd/",
            "/agencies/ebsa/",
        ]

        for link in links:
            url = response.urljoin(link)
            if url in seen or "dol.gov" not in url:
                continue
            # Only follow relevant sub-pages, limit depth
            if any(path in url for path in target_paths):
                parts = url.rstrip("/").split("/")
                if len(parts) <= 8:  # limit crawl depth
                    seen.add(url)
                    yield scrapy.Request(url, callback=self.parse_subpage)

    def parse_subpage(self, response):
        """Extract content from a DOL sub-topic page."""
        yield from self._extract_page(response)

    def _extract_page(self, response):
        """Extract policy content from a DOL page."""
        title = response.css("h1::text").get("").strip()
        if not title:
            title = response.css("title::text").get("").strip()

        content_selectors = [
            ".field--name-body",
            "#block-dol-theme-content",
            "article",
            ".main-content",
            "#content",
        ]

        text = ""
        for sel in content_selectors:
            block = response.css(sel)
            if block:
                text = " ".join(block.css("p::text, li::text, h2::text, h3::text, td::text").getall())
                break

        if not text:
            text = " ".join(response.css("p::text, li::text").getall())

        cleaned = " ".join(text.split()).strip()

        if cleaned and len(cleaned) > 100:
            yield PolicyDocument(
                title=title,
                url=response.url,
                source="DOL",
                category="labor_law",
                content=cleaned[:15000],
                scraped_at=datetime.now(timezone.utc).isoformat(),
            )
