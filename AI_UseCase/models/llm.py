"""
Multi-Provider LLM Factory
Supports Groq, xAI Grok, OpenAI, and Google Gemini via LangChain.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import ROUTER_MODEL, get_default_temperature

logger = logging.getLogger(__name__)


def get_llm(provider, model_name, api_key, temperature=None):
    """
    Factory function to create LLM instances for any supported provider.

    Args:
        provider: "groq", "openai", or "gemini"
        model_name: Model identifier string
        api_key: API key for the provider
        temperature: Sampling temperature (0.0 - 1.0)

    Returns:
        BaseChatModel instance
    """
    if temperature is None:
        temperature = get_default_temperature()

    try:
        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
            )
        elif provider == "xai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                base_url="https://api.x.ai/v1",
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
            )
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model_name,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error(f"Failed to initialize {provider} model '{model_name}': {e}")
        raise RuntimeError(f"Failed to initialize {provider} model: {str(e)}")


def get_router_llm(api_key):
    """
    Get a fast, cheap LLM for intent routing.
    Uses Groq's smallest model for near-zero latency classification.

    Args:
        api_key: Groq API key

    Returns:
        ChatGroq instance configured for fast routing
    """
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=api_key,
            model=ROUTER_MODEL,
            temperature=0.0,  # Deterministic for classification
        )
    except Exception as e:
        logger.error(f"Failed to initialize router LLM: {e}")
        raise RuntimeError(f"Failed to initialize router LLM: {str(e)}")


def get_chatgroq_model(api_key=None, model_name="llama-3.3-70b-versatile"):
    """
    Backward-compatible wrapper for the original template.
    """
    return get_llm("groq", model_name, api_key)
