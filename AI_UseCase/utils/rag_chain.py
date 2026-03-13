"""
RAG Query Pipeline
Retrieves relevant documents and generates augmented responses.
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import RAG_PROMPT_TEMPLATE, BASE_SYSTEM_PROMPT, CONCISE_PROMPT, DETAILED_PROMPT

logger = logging.getLogger(__name__)


def rag_query(query, retriever, llm, response_mode="concise", chat_history=None):
    """
    Execute a RAG query: retrieve relevant chunks, build augmented prompt, generate answer.

    Args:
        query: User's question
        retriever: Vector store retriever
        llm: LLM instance for generation
        response_mode: "concise" or "detailed"
        chat_history: List of previous messages (optional)

    Returns:
        dict with keys: answer, sources, reasoning
    """
    try:
        # Step 1: Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information in the loaded documents. "
                          "Please make sure relevant documents are uploaded, or try rephrasing your question.",
                "sources": [],
                "reasoning": "No relevant documents found in vector store.",
            }

        # Step 2: Build context from retrieved chunks
        context_parts = []
        sources = []

        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[Chunk {i+1}] {doc.page_content}")
            sources.append({
                "filename": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "chunk_index": doc.metadata.get("chunk_index", i),
            })

        context = "\n\n".join(context_parts)
        sources_str = ", ".join([f"{s['filename']} (Page {s['page']})" for s in sources])

        # Step 3: Build the augmented prompt
        rag_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            sources=sources_str,
            question=query,
        )

        # Step 4: Compose system prompt with response mode
        mode_prompt = CONCISE_PROMPT if response_mode == "concise" else DETAILED_PROMPT
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{mode_prompt}"

        # Step 5: Build message list with chat history
        messages = [SystemMessage(content=system_prompt)]

        if chat_history:
            # Include last few messages for context (sliding window)
            from langchain_core.messages import HumanMessage as HM, AIMessage as AM
            recent_history = chat_history[-6:]  # Last 3 exchanges
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HM(content=msg["content"]))
                else:
                    messages.append(AM(content=msg["content"]))

        messages.append(HumanMessage(content=rag_prompt))

        # Step 6: Generate response
        response = llm.invoke(messages)

        reasoning = (
            f"Retrieved {len(retrieved_docs)} chunks from {len(set(s['filename'] for s in sources))} "
            f"document(s). Sources: {sources_str}"
        )

        return {
            "answer": response.content,
            "sources": sources,
            "reasoning": reasoning,
        }

    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return {
            "answer": f"An error occurred while searching documents: {str(e)}",
            "sources": [],
            "reasoning": f"Error: {str(e)}",
        }


def rag_query_stream(query, retriever, llm, response_mode="concise", chat_history=None):
    """
    Streaming version of rag_query. Yields tokens as they are generated.

    Args:
        Same as rag_query

    Yields:
        dict with 'token' (str) or final 'sources' and 'reasoning'
    """
    try:
        # Step 1: Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            yield {
                "type": "complete",
                "answer": "I couldn't find any relevant information in the loaded documents.",
                "sources": [],
                "reasoning": "No relevant documents found.",
            }
            return

        # Step 2: Build context
        context_parts = []
        sources = []

        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[Chunk {i+1}] {doc.page_content}")
            sources.append({
                "filename": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "chunk_index": doc.metadata.get("chunk_index", i),
            })

        context = "\n\n".join(context_parts)
        sources_str = ", ".join([f"{s['filename']} (Page {s['page']})" for s in sources])

        # Step 3: Build prompt
        rag_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            sources=sources_str,
            question=query,
        )

        mode_prompt = CONCISE_PROMPT if response_mode == "concise" else DETAILED_PROMPT
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{mode_prompt}"

        messages = [SystemMessage(content=system_prompt)]

        if chat_history:
            from langchain_core.messages import HumanMessage as HM, AIMessage as AM
            recent_history = chat_history[-6:]
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HM(content=msg["content"]))
                else:
                    messages.append(AM(content=msg["content"]))

        messages.append(HumanMessage(content=rag_prompt))

        # Step 4: Stream response
        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_response += token
                yield {"type": "token", "token": token}

        reasoning = (
            f"Retrieved {len(retrieved_docs)} chunks from {len(set(s['filename'] for s in sources))} "
            f"document(s). Sources: {sources_str}"
        )

        yield {
            "type": "complete",
            "answer": full_response,
            "sources": sources,
            "reasoning": reasoning,
        }

    except Exception as e:
        logger.error(f"RAG stream error: {e}")
        yield {
            "type": "complete",
            "answer": f"An error occurred: {str(e)}",
            "sources": [],
            "reasoning": f"Error: {str(e)}",
        }
