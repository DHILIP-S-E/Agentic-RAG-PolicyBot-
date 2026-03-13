"""
Scrapy Settings for PolicyBot scraper.
"""

BOT_NAME = "policybot_scraper"
SPIDER_MODULES = ["scraper.spiders"]
NEWSPIDER_MODULE = "scraper.spiders"

# Be respectful — identify ourselves and throttle requests
USER_AGENT = "PolicyBot-Scraper/1.0 (+https://github.com/policybot; educational/research use)"
ROBOTSTXT_OBEY = True

# Throttle to avoid overloading government sites
CONCURRENT_REQUESTS = 4
DOWNLOAD_DELAY = 1.5
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Retry settings
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Timeout
DOWNLOAD_TIMEOUT = 30

# Disable cookies (not needed for public pages)
COOKIES_ENABLED = False

# Pipeline — Qdrant ingestion is handled by the runner script, not Scrapy pipelines
# We use feed exports to dump JSON, then ingest separately
FEEDS = {
    "scraped_data.jsonl": {
        "format": "jsonlines",
        "encoding": "utf-8",
        "overwrite": True,
    },
}

# Logging
LOG_LEVEL = "INFO"

# Request fingerprinter (Scrapy 2.7+)
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
