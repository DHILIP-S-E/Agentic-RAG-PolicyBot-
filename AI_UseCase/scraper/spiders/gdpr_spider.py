"""
GDPR Articles Spider
Scrapes all 99 GDPR articles from gdpr-info.eu
"""

import scrapy
from datetime import datetime, timezone
from scraper.items import PolicyDocument


class GdprSpider(scrapy.Spider):
    name = "gdpr"
    allowed_domains = ["gdpr-info.eu"]

    # Generate URLs for all 99 GDPR articles
    start_urls = [f"https://gdpr-info.eu/art-{i}-gdpr/" for i in range(1, 100)]

    custom_settings = {
        "DOWNLOAD_DELAY": 1.5,
    }

    def parse(self, response):
        """Extract content from a GDPR article page."""
        title = response.css("h1::text").get("").strip()
        if not title:
            title = response.css(".entry-title::text").get("").strip()

        # Article content
        content_block = response.css(".entry-content")
        if content_block:
            text = " ".join(content_block.css("p::text, li::text, ol li::text").getall())
        else:
            text = " ".join(response.css("p::text, li::text").getall())

        # Also grab the article number from URL
        article_num = response.url.split("/art-")[-1].split("-gdpr")[0] if "/art-" in response.url else ""

        cleaned = " ".join(text.split()).strip()

        if cleaned and len(cleaned) > 50:
            full_title = f"GDPR Article {article_num}: {title}" if article_num else title

            yield PolicyDocument(
                title=full_title,
                url=response.url,
                source="GDPR",
                category="data_privacy",
                content=cleaned[:15000],
                scraped_at=datetime.now(timezone.utc).isoformat(),
            )
