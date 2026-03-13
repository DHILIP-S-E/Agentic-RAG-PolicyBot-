"""
RAG Embedding Models
Supports HuggingFace (free, local) and OpenAI embeddings.
"""

import logging
import streamlit as st
from config.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(provider="huggingface"):
    """
    Load and cache the embedding model.

    Args:
        provider: "huggingface" (free, local) or "openai" (requires API key)

    Returns:
        Embeddings instance
    """
    try:
        if provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"Loaded HuggingFace embedding model: {EMBEDDING_MODEL_NAME}")
            return model

        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            model = OpenAIEmbeddings(model="text-embedding-3-small")
            logger.info("Loaded OpenAI embedding model: text-embedding-3-small")
            return model

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    except Exception as e:
        logger.error(f"Failed to load embedding model ({provider}): {e}")
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")
