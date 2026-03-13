"""
Vector Store Management
Qdrant (persistent) with ChromaDB fallback (in-memory).
"""

import logging
import uuid
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _get_qdrant_client(qdrant_url, qdrant_api_key):
    """
    Create a Qdrant client.
    Handles sslip.io domains by using direct IP + Host header patching.
    """
    from qdrant_client import QdrantClient
    import httpx

    parsed = urlparse(qdrant_url.rstrip("/"))
    hostname = parsed.hostname
    scheme = parsed.scheme or "http"
    port = parsed.port or (443 if scheme == "https" else 80)

    # For sslip.io domains, use direct IP with Host header to fix DNS resolution
    if hostname and "sslip.io" in hostname:
        parts = hostname.split(".")
        # Extract embedded IP: *.A.B.C.D.sslip.io
        sslip_idx = None
        for i, part in enumerate(parts):
            if part == "sslip":
                sslip_idx = i
                break

        if sslip_idx and sslip_idx >= 4:
            direct_ip = ".".join(parts[sslip_idx - 4:sslip_idx])
            logger.info(f"Qdrant sslip.io detected: using direct IP {direct_ip} with Host: {hostname}")

            client = QdrantClient(
                host=direct_ip,
                port=port,
                api_key=qdrant_api_key,
                timeout=30,
                prefer_grpc=False,
                https=scheme == "https",
                check_compatibility=False,
            )

            # Patch the internal httpx client to include Host header
            patched_httpx = httpx.Client(
                base_url=f"{scheme}://{direct_ip}:{port}",
                headers={
                    "Host": hostname,
                    "api-key": qdrant_api_key or "",
                },
                timeout=30,
            )
            client._client.http.client._client = patched_httpx
            return client

    # Standard connection for regular URLs
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=30,
        prefer_grpc=False,
    )
    return client


def create_vector_store(chunks, embedding_model, qdrant_url=None, qdrant_api_key=None):
    """
    Create a vector store from document chunks.
    Uses Qdrant if URL is provided, falls back to ChromaDB.

    Args:
        chunks: List of LangChain Document objects (chunked)
        embedding_model: Embeddings instance
        qdrant_url: Optional Qdrant server URL
        qdrant_api_key: Optional Qdrant API key

    Returns:
        Vector store instance
    """
    if qdrant_url:
        return _create_qdrant_store(chunks, embedding_model, qdrant_url, qdrant_api_key)
    else:
        return _create_chroma_store(chunks, embedding_model)


def _create_qdrant_store(chunks, embedding_model, qdrant_url, qdrant_api_key):
    """Create Qdrant vector store with persistent storage."""
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client.models import Distance, VectorParams

        collection_name = "policybot_docs"
        client = _get_qdrant_client(qdrant_url, qdrant_api_key)

        # Check if collection exists, create if not
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

        # Create LangChain wrapper
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        # Add documents
        if chunks:
            ids = [str(uuid.uuid4()) for _ in chunks]
            vector_store.add_documents(chunks, ids=ids)
            logger.info(f"Added {len(chunks)} chunks to Qdrant")

        return vector_store

    except Exception as e:
        logger.error(f"Qdrant error, falling back to ChromaDB: {e}")
        return _create_chroma_store(chunks, embedding_model)


def _create_chroma_store(chunks, embedding_model):
    """Create in-memory ChromaDB vector store (fallback)."""
    try:
        from langchain_chroma import Chroma

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="policybot_docs",
        )
        logger.info(f"Created ChromaDB store with {len(chunks)} chunks")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating ChromaDB store: {e}")
        raise


def get_retriever(vector_store, top_k=6):
    """
    Get a retriever from the vector store.

    Args:
        vector_store: Vector store instance
        top_k: Number of results to retrieve

    Returns:
        BaseRetriever instance
    """
    try:
        store_type = type(vector_store).__name__

        if store_type == "QdrantVectorStore":
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
        else:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "lambda_mult": 0.7},
            )

        return retriever

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise


def add_documents_to_store(vector_store, new_chunks):
    """Add new document chunks to an existing vector store."""
    try:
        ids = [str(uuid.uuid4()) for _ in new_chunks]
        vector_store.add_documents(new_chunks, ids=ids)
        logger.info(f"Added {len(new_chunks)} new chunks to vector store")

    except Exception as e:
        logger.error(f"Error adding documents to store: {e}")
        raise
