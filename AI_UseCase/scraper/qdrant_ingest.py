"""
Qdrant Ingestion Pipeline
Reads scraped JSONL data, chunks it, embeds it, and stores in Qdrant.
"""

import json
import logging
import uuid
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def load_scraped_data(jsonl_path):
    """Load scraped documents from JSONL file."""
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "source": item.get("source", ""),
                        "category": item.get("category", ""),
                        "scraped_at": item.get("scraped_at", ""),
                        "type": "scraped_web",
                    },
                )
                documents.append(doc)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping malformed line: {e}")
    logger.info(f"Loaded {len(documents)} documents from {jsonl_path}")
    return documents


def chunk_documents(documents):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def get_embedding_model():
    """Load HuggingFace embedding model (no Streamlit dependency)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    return model


def ingest_to_qdrant(chunks, qdrant_url, qdrant_api_key=None, collection_name="policybot_docs", batch_size=50):
    """
    Embed and store document chunks in Qdrant.

    Args:
        chunks: List of LangChain Document objects
        qdrant_url: Qdrant server URL
        qdrant_api_key: Optional Qdrant API key
        collection_name: Qdrant collection name
        batch_size: Number of documents to upsert per batch
    """
    from qdrant_client.models import Distance, VectorParams
    from langchain_qdrant import QdrantVectorStore
    from utils.vector_store import _get_qdrant_client

    embedding_model = get_embedding_model()

    # Connect to Qdrant (reuse sslip.io-aware client from vector_store)
    client = _get_qdrant_client(qdrant_url, qdrant_api_key)

    # Ensure collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        test_embedding = embedding_model.embed_query("test")
        vector_size = len(test_embedding)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection '{collection_name}' (dim={vector_size})")
    else:
        logger.info(f"Using existing Qdrant collection '{collection_name}'")

    # Create LangChain wrapper
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    # Ingest in batches
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        ids = [str(uuid.uuid4()) for _ in batch]
        vector_store.add_documents(batch, ids=ids)
        logger.info(f"Ingested batch {i // batch_size + 1}: {len(batch)} chunks ({i + len(batch)}/{total})")

    logger.info(f"Successfully ingested {total} chunks into Qdrant collection '{collection_name}'")
    return total
