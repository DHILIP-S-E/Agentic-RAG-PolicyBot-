"""
Core Chat Engine (Orchestrator)
Routes queries through the Agentic RAG pipeline.
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.config import BASE_SYSTEM_PROMPT, CONCISE_PROMPT, DETAILED_PROMPT
from utils.router import classify_intent
from utils.rag_chain import rag_query, rag_query_stream
from utils.web_search import web_augmented_query, web_augmented_query_stream

logger = logging.getLogger(__name__)


def process_query(query, chat_history, llm, router_llm, retriever,
                  response_mode="concise", has_web_search=False, tavily_key=None):
    """
    Main orchestrator: classifies intent and routes to the appropriate pipeline.

    Args:
        query: User's question
        chat_history: List of {"role": str, "content": str} dicts
        llm: Main LLM instance for generation
        router_llm: Fast LLM for intent classification
        retriever: Vector store retriever (or None if no docs)
        response_mode: "concise" or "detailed"
        has_web_search: Whether web search is available
        tavily_key: Optional Tavily API key

    Returns:
        dict with keys: answer, sources, route, reasoning
    """
    try:
        has_documents = retriever is not None

        # Step 1: Classify intent
        route = classify_intent(query, has_documents, has_web_search, router_llm)

        # Step 2: Route to appropriate pipeline
        if route == "RAG":
            result = rag_query(query, retriever, llm, response_mode, chat_history)
        elif route == "WEB":
            result = web_augmented_query(query, llm, response_mode, chat_history, tavily_key)
        else:  # DIRECT
            result = _direct_query(query, llm, response_mode, chat_history)

        result["route"] = route
        return result

    except Exception as e:
        logger.error(f"Chat engine error: {e}")
        return {
            "answer": f"An error occurred while processing your query: {str(e)}",
            "sources": [],
            "route": "ERROR",
            "reasoning": f"Error: {str(e)}",
        }


def process_query_stream(query, chat_history, llm, router_llm, retriever,
                         response_mode="concise", has_web_search=False, tavily_key=None):
    """
    Streaming version of process_query. Yields tokens and metadata.

    Yields:
        dicts with:
          - {"type": "route", "route": str} (first yield)
          - {"type": "token", "token": str} (streaming tokens)
          - {"type": "complete", "answer": str, "sources": list, "reasoning": str} (final)
    """
    try:
        has_documents = retriever is not None

        # Step 1: Classify intent
        route = classify_intent(query, has_documents, has_web_search, router_llm)

        # Yield the route first so UI can show badge
        yield {"type": "route", "route": route}

        # Step 2: Stream from appropriate pipeline
        if route == "RAG":
            stream = rag_query_stream(query, retriever, llm, response_mode, chat_history)
        elif route == "WEB":
            stream = web_augmented_query_stream(query, llm, response_mode, chat_history, tavily_key)
        else:  # DIRECT
            stream = _direct_query_stream(query, llm, response_mode, chat_history)

        for item in stream:
            if item.get("type") == "complete":
                item["route"] = route
            yield item

    except Exception as e:
        logger.error(f"Chat engine stream error: {e}")
        yield {
            "type": "complete",
            "answer": f"An error occurred: {str(e)}",
            "sources": [],
            "route": "ERROR",
            "reasoning": f"Error: {str(e)}",
        }


def _direct_query(query, llm, response_mode="concise", chat_history=None):
    """
    Direct LLM response without RAG or web search.
    """
    try:
        mode_prompt = CONCISE_PROMPT if response_mode == "concise" else DETAILED_PROMPT
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{mode_prompt}"

        messages = [SystemMessage(content=system_prompt)]

        if chat_history:
            recent_history = chat_history[-10:]  # Last 5 exchanges
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=query))

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": [],
            "reasoning": "Answered directly from LLM knowledge (no document retrieval or web search needed).",
        }

    except Exception as e:
        logger.error(f"Direct query error: {e}")
        return {
            "answer": f"An error occurred: {str(e)}",
            "sources": [],
            "reasoning": f"Error: {str(e)}",
        }


def _direct_query_stream(query, llm, response_mode="concise", chat_history=None):
    """
    Streaming version of direct LLM query.
    """
    try:
        mode_prompt = CONCISE_PROMPT if response_mode == "concise" else DETAILED_PROMPT
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{mode_prompt}"

        messages = [SystemMessage(content=system_prompt)]

        if chat_history:
            recent_history = chat_history[-10:]
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=query))

        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_response += token
                yield {"type": "token", "token": token}

        yield {
            "type": "complete",
            "answer": full_response,
            "sources": [],
            "reasoning": "Answered directly from LLM knowledge.",
        }

    except Exception as e:
        logger.error(f"Direct stream error: {e}")
        yield {
            "type": "complete",
            "answer": f"An error occurred: {str(e)}",
            "sources": [],
            "reasoning": f"Error: {str(e)}",
        }
