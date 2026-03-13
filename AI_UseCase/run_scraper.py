#!/usr/bin/env python3
"""
PolicyBot Scraper Runner
Scrapes realtime policy/compliance data from government sources and ingests into Qdrant.

Usage:
    # Scrape all sources and ingest into Qdrant
    python run_scraper.py --qdrant-url http://localhost:6333

    # Scrape specific spider only
    python run_scraper.py --spider osha_topics --qdrant-url http://localhost:6333

    # Scrape only (no Qdrant ingestion) — saves to scraped_data.jsonl
    python run_scraper.py --scrape-only

    # Ingest only (from existing JSONL file)
    python run_scraper.py --ingest-only --qdrant-url http://localhost:6333

Available spiders: osha_topics, eeoc, gdpr, dol
"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_scraper")

SPIDERS = ["osha_topics", "eeoc", "gdpr", "dol"]
JSONL_FILE = os.path.join(os.path.dirname(__file__), "scraped_data.jsonl")


def run_spiders(spider_names):
    """Run Scrapy spiders and output to JSONL."""
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    # Clean previous output
    if os.path.exists(JSONL_FILE):
        os.remove(JSONL_FILE)

    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "scraper.settings")
    settings = get_project_settings()
    settings.set("FEEDS", {
        JSONL_FILE: {
            "format": "jsonlines",
            "encoding": "utf-8",
            "overwrite": True,
        },
    })

    process = CrawlerProcess(settings)

    for spider_name in spider_names:
        logger.info(f"Queuing spider: {spider_name}")
        process.crawl(spider_name)

    logger.info(f"Starting {len(spider_names)} spider(s)...")
    process.start()
    logger.info("All spiders finished.")

    # Report stats
    if os.path.exists(JSONL_FILE):
        with open(JSONL_FILE, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        logger.info(f"Scraped {count} documents → {JSONL_FILE}")
        return count
    return 0


def run_ingestion(qdrant_url, qdrant_api_key=None, collection_name="policybot_docs"):
    """Ingest scraped JSONL data into Qdrant."""
    from scraper.qdrant_ingest import load_scraped_data, chunk_documents, ingest_to_qdrant

    if not os.path.exists(JSONL_FILE):
        logger.error(f"No scraped data found at {JSONL_FILE}. Run scraping first.")
        return 0

    logger.info(f"Loading scraped data from {JSONL_FILE}...")
    documents = load_scraped_data(JSONL_FILE)

    if not documents:
        logger.warning("No documents loaded. Nothing to ingest.")
        return 0

    logger.info(f"Chunking {len(documents)} documents...")
    chunks = chunk_documents(documents)

    logger.info(f"Ingesting {len(chunks)} chunks into Qdrant at {qdrant_url}...")
    total = ingest_to_qdrant(
        chunks,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
    )
    return total


def main():
    parser = argparse.ArgumentParser(
        description="PolicyBot Scraper — Scrape policy data and ingest into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--spider", type=str, default=None,
        help=f"Run a specific spider. Options: {', '.join(SPIDERS)}",
    )
    parser.add_argument(
        "--qdrant-url", type=str, default=os.environ.get("QDRANT_URL", ""),
        help="Qdrant server URL (e.g. http://localhost:6333)",
    )
    parser.add_argument(
        "--qdrant-api-key", type=str, default=os.environ.get("QDRANT_API_KEY", ""),
        help="Qdrant API key",
    )
    parser.add_argument(
        "--collection", type=str, default="policybot_docs",
        help="Qdrant collection name (default: policybot_docs)",
    )
    parser.add_argument(
        "--scrape-only", action="store_true",
        help="Only scrape, don't ingest into Qdrant",
    )
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="Only ingest existing JSONL into Qdrant (skip scraping)",
    )

    args = parser.parse_args()

    # Determine which spiders to run
    spider_names = [args.spider] if args.spider else SPIDERS

    # Validate spider names
    for s in spider_names:
        if s not in SPIDERS:
            logger.error(f"Unknown spider: {s}. Available: {', '.join(SPIDERS)}")
            sys.exit(1)

    # Step 1: Scrape
    if not args.ingest_only:
        logger.info("=" * 60)
        logger.info("STEP 1: SCRAPING POLICY DATA")
        logger.info("=" * 60)
        doc_count = run_spiders(spider_names)
        if doc_count == 0:
            logger.warning("No documents scraped. Check your internet connection.")
            if not args.scrape_only:
                sys.exit(1)

    # Step 2: Ingest into Qdrant
    if not args.scrape_only:
        if not args.qdrant_url:
            logger.error("Qdrant URL required for ingestion. Use --qdrant-url or set QDRANT_URL env var.")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("STEP 2: INGESTING INTO QDRANT")
        logger.info("=" * 60)
        chunk_count = run_ingestion(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key or None,
            collection_name=args.collection,
        )
        logger.info(f"Done! {chunk_count} chunks ingested into Qdrant.")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
