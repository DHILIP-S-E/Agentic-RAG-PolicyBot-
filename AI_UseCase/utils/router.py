"""
Agentic Intent Router
Classifies user queries into RAG, WEB, or DIRECT routes.
"""

import logging
from langchain_core.messages import HumanMessage
from config.config import ROUTER_PROMPT

logger = logging.getLogger(__name__)


def classify_intent(query, has_documents, has_web_search, router_llm):
    """
    Classify user query intent using a fast LLM call.

    Args:
        query: User's question
        has_documents: Whether documents are loaded in the vector store
        has_web_search: Whether web search is available
        router_llm: Fast LLM instance for classification

    Returns:
        str: "RAG", "WEB", or "DIRECT"
    """
    try:
        # Apply constraints before calling LLM
        # If no documents loaded, RAG is not an option
        if not has_documents and not has_web_search:
            return "DIRECT"

        # Format the router prompt
        prompt = ROUTER_PROMPT.format(question=query)

        # Call the router LLM
        response = router_llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().upper()

        # Parse the response - extract the classification
        for valid_intent in ["RAG", "WEB", "DIRECT"]:
            if valid_intent in intent:
                intent = valid_intent
                break
        else:
            # Default to DIRECT if response is unclear
            intent = "DIRECT"

        # Apply post-classification constraints
        if intent == "RAG" and not has_documents:
            intent = "WEB" if has_web_search else "DIRECT"

        if intent == "WEB" and not has_web_search:
            intent = "RAG" if has_documents else "DIRECT"

        logger.info(f"Query classified as: {intent} | Query: {query[:50]}...")
        return intent

    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        # Fallback: if docs available use RAG, else DIRECT
        if has_documents:
            return "RAG"
        return "DIRECT"
