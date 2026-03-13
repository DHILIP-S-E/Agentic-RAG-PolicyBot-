"""
OSHA Safety & Health Topics Spider
Scrapes workplace safety topics from osha.gov/topics
"""

import scrapy
from datetime import datetime, timezone
from scraper.items import PolicyDocument


class OshaTopicsSpider(scrapy.Spider):
    name = "osha_topics"
    allowed_domains = ["osha.gov"]
    start_urls = ["https://www.osha.gov/topics"]

    custom_settings = {
        "DOWNLOAD_DELAY": 2,
    }

    def parse(self, response):
        """Parse the A-Z topics index page and follow each topic link."""
        topic_links = response.css("a[href*='/topics/']::attr(href)").getall()
        seen = set()
        for link in topic_links:
            url = response.urljoin(link)
            # Only follow /topics/{name} paths, skip deep sub-pages
            parts = url.rstrip("/").split("/")
            if len(parts) >= 5 and parts[3] == "topics" and url not in seen:
                seen.add(url)
                yield scrapy.Request(url, callback=self.parse_topic)

    def parse_topic(self, response):
        """Extract content from an individual OSHA topic page."""
        title = response.css("h1::text").get("").strip()
        if not title:
            title = response.css("title::text").get("").strip()

        # Main content area
        content_selectors = [
            "#main-content",
            ".field--name-body",
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
                source="OSHA",
                category="workplace_safety",
                content=cleaned[:15000],
                scraped_at=datetime.now(timezone.utc).isoformat(),
            )
