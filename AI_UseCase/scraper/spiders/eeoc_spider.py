"""
EEOC Laws & Guidance Spider
Scrapes employment discrimination laws and guidance from eeoc.gov
"""

import scrapy
from datetime import datetime, timezone
from scraper.items import PolicyDocument


class EeocSpider(scrapy.Spider):
    name = "eeoc"
    allowed_domains = ["eeoc.gov"]
    start_urls = [
        "https://www.eeoc.gov/laws-guidance",
        "https://www.eeoc.gov/employers",
        "https://www.eeoc.gov/laws/types",
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 2,
    }

    def parse(self, response):
        """Parse index pages and follow links to guidance/law pages."""
        # Follow links to individual law/guidance pages
        links = response.css("a[href]::attr(href)").getall()
        seen = set()

        target_paths = [
            "/laws/guidance/",
            "/laws/statutes/",
            "/laws/types/",
            "/employers/",
            "/harassment",
            "/retaliation",
            "/disability-discrimination",
            "/age-discrimination",
            "/race-discrimination",
            "/sex-based-discrimination",
            "/national-origin-discrimination",
            "/religious-discrimination",
            "/pregnancy-discrimination",
            "/equal-pay",
            "/genetic-information-discrimination",
        ]

        for link in links:
            url = response.urljoin(link)
            if url in seen or "eeoc.gov" not in url:
                continue
            if any(path in url for path in target_paths):
                seen.add(url)
                yield scrapy.Request(url, callback=self.parse_page)

    def parse_page(self, response):
        """Extract content from an EEOC guidance/law page."""
        title = response.css("h1::text").get("").strip()
        if not title:
            title = response.css("title::text").get("").strip()

        content_selectors = [
            ".field--name-body",
            "#block-eeoc-content",
            "article",
            ".main-content",
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
                source="EEOC",
                category="employment_law",
                content=cleaned[:15000],
                scraped_at=datetime.now(timezone.utc).isoformat(),
            )
