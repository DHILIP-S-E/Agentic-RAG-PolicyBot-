"""
PolicyBot - Central Configuration
All API keys, settings, prompts, and model options.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

def get_default_provider():
    return os.environ.get("LLM_PROVIDER", "groq")

def get_default_model():
    return os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

def get_default_temperature():
    return float(os.environ.get("LLM_TEMPERATURE", "0.7"))

# =============================================================================
# MODEL OPTIONS (per provider)
# =============================================================================

MODEL_OPTIONS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "xai": [
        "grok-3-mini-fast",
        "grok-3-fast",
        "grok-2",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
}

# Router uses the cheapest/fastest model
ROUTER_MODEL = "llama-3.1-8b-instant"

# =============================================================================
# RAG SETTINGS
# =============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 6
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# =============================================================================
# WEB SEARCH SETTINGS
# =============================================================================

TAVILY_MAX_RESULTS = 3
DDG_MAX_RESULTS = 3

# =============================================================================
# QDRANT SETTINGS
# =============================================================================

QDRANT_COLLECTION_NAME = "policybot_docs"

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

BASE_SYSTEM_PROMPT = """You are PolicyBot, a Corporate Policy Intelligence Assistant.
You help employees instantly understand their organization's policies, HR guidelines, compliance requirements, and workplace procedures.

Important rules for your responses:
- NEVER mention your internal retrieval process, RAG pipeline, embeddings, or vector databases.
- NEVER say things like "The documents do not define..." or "Based on my retrieval..." or "I don't have specific information..."
- NEVER refuse to answer. You ALWAYS have something useful to say.
- If you are given context (documents or web results), you MUST use that context to answer. Do NOT ignore provided context.
- When answering from documents, reference them naturally: "According to the Employee Handbook..." or "Your company's leave policy states..."
- When answering from web results, synthesize the information into a helpful answer and cite sources by name.
- Always be helpful, professional, and actionable.
- After answering, suggest a relevant follow-up question when appropriate."""

CONCISE_PROMPT = """Respond concisely in 2-3 sentences maximum. Use bullet points for clarity.
Get straight to the answer — no preamble, no filler."""

DETAILED_PROMPT = """Provide a comprehensive, well-structured response:
- Start with a clear direct answer
- Add context and relevant details
- Use bullet points or numbered lists for clarity
- Include specific references to source documents naturally
- End with a helpful suggestion or follow-up if appropriate
Use clear headings and formatting to make the response easy to scan."""

# =============================================================================
# RAG PROMPT TEMPLATE
# =============================================================================

RAG_PROMPT_TEMPLATE = """Answer the user's question using the company document excerpts below.
Speak as a knowledgeable policy advisor — never mention "retrieved context", "chunks", or "vector search".
Reference documents naturally: "According to [Document Name]..." or "Your [Policy Name] states..."

If the excerpts don't fully answer the question, share what you can and suggest the user upload more specific documents.

---
{context}

Source documents: {sources}
---

Question: {question}

Provide a clear, actionable answer. If relevant, suggest a follow-up question the user might want to ask."""

# =============================================================================
# WEB SEARCH PROMPT TEMPLATE
# =============================================================================

WEB_SEARCH_PROMPT_TEMPLATE = """You MUST answer the user's question using the web search results provided below.
Do NOT say you don't have information — the search results below ARE your information. Use them.
Synthesize the information from these sources into one clear, helpful answer.
Cite sources naturally by their title names.

Web search results:
{search_results}

Question: {question}

IMPORTANT: Provide a direct, helpful answer based on the search results above. Never say "I don't have information" — you DO have the search results."""

# =============================================================================
# ROUTER PROMPT
# =============================================================================

ROUTER_PROMPT = """You are a query intent classifier. Given the user's question, classify it into exactly one category.

Available categories:
- RAG: The question is specifically about internal/uploaded documents, company policies, compliance rules, or specific document content that should be looked up in the knowledge base.
- WEB: The question requires real-time information, recent news, current events, live data, or information that changes frequently.
- DIRECT: Greetings, casual conversation, general knowledge questions, or anything that does NOT need document lookup or web search.

Rules (follow strictly):
1. Greetings like "hi", "hello", "hey", "good morning", "thanks", "bye" → ALWAYS DIRECT
2. Casual or vague messages that are not a specific question → ALWAYS DIRECT
3. If the question mentions "latest", "current", "today", "recent", "news" → WEB
4. Only use RAG when the user is clearly asking about specific policy, compliance, or document content.
5. When in doubt, prefer DIRECT over RAG.

Respond with ONLY one word: RAG, WEB, or DIRECT

User question: {question}"""

# =============================================================================
# API KEY HELPER
# =============================================================================

def get_api_key(provider):
    """
    Get API key with fallback chain: env var -> st.secrets -> None
    """
    key_mapping = {
        "groq": "GROQ_API_KEY",
        "xai": "XAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "qdrant": "QDRANT_API_KEY",
    }

    env_var = key_mapping.get(provider, "")

    # Try environment variable first
    value = os.environ.get(env_var)
    if value:
        return value

    # Try streamlit secrets (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if env_var in st.secrets:
            return st.secrets[env_var]
    except Exception:
        pass

    return None
