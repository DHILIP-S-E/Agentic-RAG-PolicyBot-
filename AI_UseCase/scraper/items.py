"""
Scrapy Items for PolicyBot web scraping pipeline.

Legal Note:
- US government sites (osha.gov, dol.gov, eeoc.gov) are public domain under 17 U.S.C. § 105
- GDPR text is EU public legislation, freely reproducible
- All spiders obey robots.txt and use polite rate limiting
"""

import scrapy


class PolicyDocument(scrapy.Item):
    """Represents a scraped policy/compliance document."""
    title = scrapy.Field()
    url = scrapy.Field()
    source = scrapy.Field()       # e.g. "OSHA", "EEOC", "GDPR", "DOL"
    category = scrapy.Field()     # e.g. "workplace_safety", "discrimination", "data_privacy"
    content = scrapy.Field()      # cleaned text content
    scraped_at = scrapy.Field()   # ISO timestamp
