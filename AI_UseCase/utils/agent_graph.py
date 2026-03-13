"""
Corrective RAG (CRAG) Agent Graph — LangGraph StateGraph

Production-grade agentic workflow with 8 nodes:

  START → classify_intent
    ├─ DIRECT → generate_answer → END
    ├─ WEB → web_search → generate_answer → hallucination_check → END
    └─ RAG → reformulate_query → retrieve_documents → grade_documents
                ├─ RELEVANT → generate_answer → hallucination_check → END
                └─ NOT RELEVANT → web_fallback_search → generate_answer → hallucination_check → END

Key features:
  - Intent routing (RAG / WEB / DIRECT)
  - Query reformulation using chat history (handles follow-ups)
  - Document retrieval from Qdrant vector store
  - LLM-based relevance grading (Corrective RAG)
  - Adaptive fallback: if docs aren't relevant, auto-searches web
  - Hallucination guard: verifies answer is grounded in context
  - Full reasoning trace for transparency
"""

import logging
import time
from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.config import (
    BASE_SYSTEM_PROMPT, CONCISE_PROMPT, DETAILED_PROMPT,
    RAG_PROMPT_TEMPLATE, WEB_SEARCH_PROMPT_TEMPLATE, ROUTER_PROMPT,
    TAVILY_MAX_RESULTS, DDG_MAX_RESULTS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS FOR GRAPH NODES
# =============================================================================

REFORMULATE_PROMPT = """Given the chat history and a follow-up question, rewrite the question to be a standalone, self-contained query that captures the full intent.

If the question is already standalone (not a follow-up), return it as-is.

Chat history:
{chat_history}

Follow-up question: {question}

Rewritten standalone question:"""

GRADER_PROMPT = """You are a strict document relevance grader. Given a user question and a document chunk, decide if the document ACTUALLY helps answer the specific question asked.

Rules:
- Respond YES only if the document contains specific information that directly answers or addresses the user's question.
- Respond NO if the document is only vaguely related to the same broad topic but doesn't actually help answer the question.
- If the user asks about a specific company, product, or policy name, the document must be about THAT specific entity.
- A document about "general remote work policy" is NOT relevant if the user asks about "Oppo company policy".

Respond with ONLY one word: YES or NO

User question: {question}

Document chunk:
{document}"""

HALLUCINATION_PROMPT = """You are a hallucination checker. Given a context (documents and/or web results) and an AI-generated answer, determine if the answer is faithfully grounded in the provided context.

Rules:
- If the answer accurately reflects the context without making up facts, respond GROUNDED.
- If the answer contains claims not supported by the context, respond NOT_GROUNDED.
- Minor paraphrasing is acceptable. Only flag clear fabrications.

Context:
{context}

AI Answer:
{answer}

Respond with ONLY one word: GROUNDED or NOT_GROUNDED"""

HYBRID_PROMPT = """You MUST answer the user's question using ALL the information provided below.
Do NOT say you don't have information — use the documents and web results to give a complete answer.
Prioritize company documents for policy-specific details, and supplement with web information for broader context.
Speak as a knowledgeable policy advisor. Never mention internal system processes.

---
Company Documents:
{doc_context}

Web Information:
{web_context}

Source documents: {doc_sources}
---

Question: {question}

Provide a clear, comprehensive answer that naturally references your sources. Suggest a follow-up question if relevant."""


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State passed between all graph nodes."""
    # Input
    query: str                          # original user query
    reformulated_query: str             # rewritten query (after reformulation)
    chat_history: List[dict]
    response_mode: str
    # Routing
    route: str                          # RAG, WEB, DIRECT, RAG+WEB
    # Retrieval
    retrieved_docs: List[dict]
    doc_context: str
    doc_sources: List[dict]
    doc_sources_str: str
    # Grading
    docs_relevant: bool
    relevance_score: float              # 0.0 to 1.0
    # Web search
    web_results: List[dict]
    web_context: str
    # Generation
    answer: str
    sources: List[dict]
    # Hallucination check
    is_grounded: bool
    confidence: str                     # HIGH, MEDIUM, LOW
    # Reasoning trace
    reasoning: str
    reasoning_steps: List[str]
    node_timings: List[dict]            # [{node, duration_ms}]
    # Dependencies (injected once)
    llm: object
    router_llm: object
    retriever: object
    has_web_search: bool
    tavily_key: Optional[str]


# =============================================================================
# TIMING HELPER
# =============================================================================

def _timed_step(state, node_name, step_msg):
    """Add a reasoning step with timing info."""
    steps = list(state.get("reasoning_steps", []))
    timings = list(state.get("node_timings", []))
    steps.append(step_msg)
    return steps, timings


# =============================================================================
# NODE 1: CLASSIFY INTENT
# =============================================================================

def classify_intent(state: AgentState) -> dict:
    """Route the query: RAG, WEB, or DIRECT."""
    t0 = time.time()
    query = state["query"]
    has_documents = state["retriever"] is not None
    has_web_search = state["has_web_search"]
    router_llm = state["router_llm"]

    try:
        # --- Hard pre-filter: catch greetings/casual before calling LLM ---
        query_lower = query.strip().lower().rstrip("!?.,")
        DIRECT_PATTERNS = {
            "hi", "hello", "hey", "yo", "hola", "sup", "hii", "hiii",
            "good morning", "good afternoon", "good evening", "good night",
            "thanks", "thank you", "thx", "bye", "goodbye", "see you",
            "how are you", "whats up", "what's up", "howdy",
            "ok", "okay", "sure", "yes", "no", "cool", "nice",
        }
        if query_lower in DIRECT_PATTERNS or len(query_lower) <= 3:
            ms = int((time.time() - t0) * 1000)
            return {
                "route": "DIRECT",
                "reasoning_steps": [f"[{ms}ms] Classify Intent → DIRECT (greeting/casual)"],
                "node_timings": [{"node": "classify_intent", "duration_ms": ms}],
            }

        if not has_documents and not has_web_search:
            ms = int((time.time() - t0) * 1000)
            return {
                "route": "DIRECT",
                "reasoning_steps": [f"[{ms}ms] Classify Intent → DIRECT (no docs/web)"],
                "node_timings": [{"node": "classify_intent", "duration_ms": ms}],
            }

        prompt = ROUTER_PROMPT.format(question=query)
        response = router_llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().upper()

        for valid in ["RAG", "WEB", "DIRECT"]:
            if valid in intent:
                intent = valid
                break
        else:
            intent = "DIRECT"

        # Constraint enforcement
        if intent == "RAG" and not has_documents:
            intent = "WEB" if has_web_search else "DIRECT"
        if intent == "WEB" and not has_web_search:
            intent = "RAG" if has_documents else "DIRECT"

        ms = int((time.time() - t0) * 1000)
        logger.info(f"Router: '{query[:50]}' → {intent} ({ms}ms)")
        return {
            "route": intent,
            "reasoning_steps": [f"[{ms}ms] Classify Intent → {intent}"],
            "node_timings": [{"node": "classify_intent", "duration_ms": ms}],
        }

    except Exception as e:
        logger.error(f"Router error: {e}")
        fallback = "RAG" if has_documents else "DIRECT"
        return {
            "route": fallback,
            "reasoning_steps": [f"Classify Intent error → fallback {fallback}"],
            "node_timings": [{"node": "classify_intent", "duration_ms": 0}],
        }


# =============================================================================
# NODE 2: REFORMULATE QUERY (handles follow-ups)
# =============================================================================

def reformulate_query(state: AgentState) -> dict:
    """Rewrite follow-up questions into standalone queries using chat history."""
    t0 = time.time()
    query = state["query"]
    chat_history = state.get("chat_history", [])
    router_llm = state["router_llm"]
    steps = list(state.get("reasoning_steps", []))
    timings = list(state.get("node_timings", []))

    # Skip reformulation if no chat history or query is already clear
    if not chat_history or len(chat_history) < 2:
        ms = int((time.time() - t0) * 1000)
        steps.append(f"[{ms}ms] Reformulate → skipped (no history)")
        timings.append({"node": "reformulate_query", "duration_ms": ms})
        return {"reformulated_query": query, "reasoning_steps": steps, "node_timings": timings}

    try:
        # Format recent history for the prompt
        history_str = ""
        recent = chat_history[-6:]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content'][:200]}\n"

        prompt = REFORMULATE_PROMPT.format(chat_history=history_str, question=query)
        response = router_llm.invoke([HumanMessage(content=prompt)])
        reformulated = response.content.strip()

        # Sanity check: if reformulated is empty or too different, keep original
        if not reformulated or len(reformulated) > len(query) * 3:
            reformulated = query

        ms = int((time.time() - t0) * 1000)
        if reformulated != query:
            steps.append(f"[{ms}ms] Reformulate → \"{reformulated[:80]}...\"")
        else:
            steps.append(f"[{ms}ms] Reformulate → query unchanged")

        timings.append({"node": "reformulate_query", "duration_ms": ms})
        logger.info(f"Reformulated: '{query[:40]}' → '{reformulated[:40]}'")
        return {"reformulated_query": reformulated, "reasoning_steps": steps, "node_timings": timings}

    except Exception as e:
        logger.error(f"Reformulation error: {e}")
        steps.append(f"Reformulate error → using original query")
        timings.append({"node": "reformulate_query", "duration_ms": 0})
        return {"reformulated_query": query, "reasoning_steps": steps, "node_timings": timings}


# =============================================================================
# NODE 3: RETRIEVE DOCUMENTS
# =============================================================================

def retrieve_documents(state: AgentState) -> dict:
    """Retrieve relevant chunks from the vector store."""
    t0 = time.time()
    # Use reformulated query if available, otherwise original
    query = state.get("reformulated_query") or state["query"]
    retriever = state["retriever"]
    steps = list(state.get("reasoning_steps", []))
    timings = list(state.get("node_timings", []))

    try:
        docs = retriever.invoke(query)
        ms = int((time.time() - t0) * 1000)
        steps.append(f"[{ms}ms] Retrieve → {len(docs)} chunks from vector store")
        timings.append({"node": "retrieve_documents", "duration_ms": ms})

        if not docs:
            return {
                "retrieved_docs": [], "doc_context": "",
                "doc_sources": [], "doc_sources_str": "",
                "docs_relevant": False, "reasoning_steps": steps, "node_timings": timings,
            }

        context_parts = []
        sources = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[Chunk {i+1}] {doc.page_content}")
            sources.append({
                "filename": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "excerpt": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                "chunk_index": doc.metadata.get("chunk_index", i),
            })

        context = "\n\n".join(context_parts)
        sources_str = ", ".join([f"{s['filename']} (Page {s['page']})" for s in sources])

        return {
            "retrieved_docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            "doc_context": context,
            "doc_sources": sources,
            "doc_sources_str": sources_str,
            "reasoning_steps": steps,
            "node_timings": timings,
        }

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        steps.append(f"Retrieve error: {e}")
        timings.append({"node": "retrieve_documents", "duration_ms": 0})
        return {
            "retrieved_docs": [], "doc_context": "",
            "doc_sources": [], "doc_sources_str": "",
            "docs_relevant": False, "reasoning_steps": steps, "node_timings": timings,
        }


# =============================================================================
# NODE 4: GRADE DOCUMENTS (Corrective RAG)
# =============================================================================

def grade_documents(state: AgentState) -> dict:
    """
    LLM-based relevance grading — the core of Corrective RAG.

    Three outcomes based on relevance score:
      - HIGH (>= 0.7): docs fully answer → use docs only
      - PARTIAL (0.2 - 0.7): docs partially relevant → combine docs + web (hybrid)
      - LOW (< 0.2): docs not relevant → web search only
    """
    t0 = time.time()
    query = state.get("reformulated_query") or state["query"]
    retrieved_docs = state.get("retrieved_docs", [])
    router_llm = state["router_llm"]
    steps = list(state.get("reasoning_steps", []))
    timings = list(state.get("node_timings", []))

    if not retrieved_docs:
        steps.append("Grade → no docs to grade → web fallback")
        timings.append({"node": "grade_documents", "duration_ms": 0})
        return {"docs_relevant": False, "relevance_score": 0.0,
                "reasoning_steps": steps, "node_timings": timings}

    try:
        relevant_count = 0
        total = len(retrieved_docs)

        for doc in retrieved_docs:
            content = doc.get("page_content", "")
            prompt = GRADER_PROMPT.format(question=query, document=content[:500])
            response = router_llm.invoke([HumanMessage(content=prompt)])
            grade = response.content.strip().upper()
            if "YES" in grade:
                relevant_count += 1

        relevance_ratio = relevant_count / total if total > 0 else 0

        # Three-tier grading
        if relevance_ratio >= 0.7:
            is_relevant = True
            verdict = "HIGH RELEVANCE → docs only"
        elif relevance_ratio >= 0.2:
            is_relevant = True  # Keep docs, but also add web
            verdict = f"PARTIAL ({relevance_ratio:.0%}) → docs + web supplement"
        else:
            is_relevant = False
            verdict = f"LOW ({relevance_ratio:.0%}) → web fallback"

        ms = int((time.time() - t0) * 1000)
        steps.append(f"[{ms}ms] Grade → {relevant_count}/{total} relevant ({relevance_ratio:.0%}) → {verdict}")
        timings.append({"node": "grade_documents", "duration_ms": ms})

        logger.info(f"Grader: {relevant_count}/{total} ({relevance_ratio:.0%}) → {verdict}")
        return {"docs_relevant": is_relevant, "relevance_score": relevance_ratio,
                "reasoning_steps": steps, "node_timings": timings}

    except Exception as e:
        logger.error(f"Grader error: {e}")
        steps.append(f"Grade error → using docs + web supplement")
        timings.append({"node": "grade_documents", "duration_ms": 0})
        return {"docs_relevant": True, "relevance_score": 0.3,
                "reasoning_steps": steps, "node_timings": timings}


# =============================================================================
# NODE 5: WEB SEARCH
# =============================================================================

def web_search(state: AgentState) -> dict:
    """Search the web for real-time information."""
    t0 = time.time()
    query = state.get("reformulated_query") or state["query"]
    tavily_key = state.get("tavily_key")
    steps = list(state.get("reasoning_steps", []))
    timings = list(state.get("node_timings", []))

    try:
        results = _do_web_search(query, tavily_key)
        ms = int((time.time() - t0) * 1000)
        steps.append(f"[{ms}ms] Web Search → {len(results)} results")
        timings.append({"node": "web_search", "duration_ms": ms})

        if not results:
            return {"web_results": [], "web_context": "",
                    "reasoning_steps": steps, "node_timings": timings}

        formatted = ""
        for i, r in enumerate(results):
            formatted += (
                f"\n[Source {i+1}] {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Content: {r['snippet']}\n"
            )

        return {"web_results": results, "web_context": formatted,
                "reasoning_steps": steps, "node_timings": timings}

    except Exception as e:
        logger.error(f"Web search error: {e}")
        steps.append(f"Web Search error: {e}")
        timings.append({"node": "web_search", "duration_ms": 0})
        return {"web_results": [], "web_context": "",
                "reasoning_steps": steps, "node_timings": timings}


# =============================================================================
# NODE 6: WEB FALLBACK (after RAG docs fail grading)
# =============================================================================

def web_supplement_search(state: AgentState) -> dict:
    """Supplement partial doc results with web search (keeps doc context for hybrid answer)."""
    steps = list(state.get("reasoning_steps", []))
    steps.append("Corrective action → docs partially relevant, supplementing with web")
    # Use reformulated query for better web search
    search_query = _make_web_query(state)
    result = web_search({**state, "reformulated_query": search_query, "reasoning_steps": steps})
    result["route"] = "RAG+WEB"
    return result


def web_fallback_search(state: AgentState) -> dict:
    """Full web fallback — docs are irrelevant, search web and discard doc context."""
    steps = list(state.get("reasoning_steps", []))
    steps.append("Corrective action → docs not relevant, searching web instead")
    # Use reformulated query for better web search
    search_query = _make_web_query(state)
    result = web_search({**state, "reformulated_query": search_query, "reasoning_steps": steps})
    result["route"] = "RAG+WEB"
    result["doc_context"] = ""
    result["doc_sources"] = []
    result["doc_sources_str"] = ""
    return result


def _make_web_query(state):
    """
    Make the query more specific for web search.
    E.g. 'what is policy' → 'what is corporate policy definition and examples'
    """
    query = state.get("reformulated_query") or state["query"]
    # If query is very short/vague, append context to get better web results
    if len(query.split()) <= 5:
        query = query.rstrip("?.,! ") + " corporate policy details"
    return query


# =============================================================================
# NODE 7: GENERATE ANSWER
# =============================================================================

def generate_answer(state: AgentState) -> dict:
    """Final answer generation using all available context."""
    t0 = time.time()
    query = state.get("reformulated_query") or state["query"]
    llm = state["llm"]
    response_mode = state["response_mode"]
    chat_history = state["chat_history"]
    route = state.get("route", "DIRECT")
    steps = list(state.get("reasoning_steps", []))
    timings = list(state.get("node_timings", []))

    doc_context = state.get("doc_context", "")
    doc_sources = state.get("doc_sources", [])
    doc_sources_str = state.get("doc_sources_str", "")
    web_context = state.get("web_context", "")
    web_results = state.get("web_results", [])

    try:
        if doc_context and web_context:
            user_prompt = HYBRID_PROMPT.format(
                doc_context=doc_context, web_context=web_context,
                doc_sources=doc_sources_str, question=query,
            )
            all_sources = doc_sources + web_results
            gen_type = "hybrid (docs + web)"
        elif doc_context:
            user_prompt = RAG_PROMPT_TEMPLATE.format(
                context=doc_context, sources=doc_sources_str, question=query,
            )
            all_sources = doc_sources
            gen_type = "RAG context"
        elif web_context:
            user_prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
                search_results=web_context, question=query,
            )
            all_sources = web_results
            gen_type = "web context"
        else:
            user_prompt = query
            all_sources = []
            gen_type = "direct LLM knowledge"

        messages = _build_messages(response_mode, chat_history, user_prompt)
        response = llm.invoke(messages)

        ms = int((time.time() - t0) * 1000)
        steps.append(f"[{ms}ms] Generate → {gen_type}")
        timings.append({"node": "generate_answer", "duration_ms": ms})

        return {
            "answer": response.content,
            "sources": all_sources,
            "reasoning_steps": steps,
            "node_timings": timings,
        }

    except Exception as e:
        logger.error(f"Generate error: {e}")
        steps.append(f"Generate error: {e}")
        timings.append({"node": "generate_answer", "duration_ms": 0})
        return {
            "answer": f"An error occurred: {str(e)}",
            "sources": [], "reasoning_steps": steps, "node_timings": timings,
        }


# =============================================================================
# NODE 8: HALLUCINATION CHECK
# =============================================================================

def hallucination_check(state: AgentState) -> dict:
    """Verify the generated answer is grounded in the provided context."""
    t0 = time.time()
    answer = state.get("answer", "")
    route = state.get("route", "DIRECT")
    router_llm = state["router_llm"]
    steps = list(state.get("reasoning_steps", []))
    timings = list(state.get("node_timings", []))

    # Skip hallucination check for DIRECT answers (no context to verify against)
    if route == "DIRECT" or (not state.get("doc_context") and not state.get("web_context")):
        ms = int((time.time() - t0) * 1000)
        steps.append(f"[{ms}ms] Hallucination Check → skipped (direct answer)")
        timings.append({"node": "hallucination_check", "duration_ms": ms})
        return {
            "is_grounded": True, "confidence": "MEDIUM",
            "reasoning": " → ".join(steps),
            "reasoning_steps": steps, "node_timings": timings,
        }

    try:
        # Combine all available context
        context = ""
        if state.get("doc_context"):
            context += f"Documents:\n{state['doc_context']}\n\n"
        if state.get("web_context"):
            context += f"Web Results:\n{state['web_context']}\n\n"

        prompt = HALLUCINATION_PROMPT.format(context=context[:3000], answer=answer[:1500])
        response = router_llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip().upper()

        is_grounded = "GROUNDED" in result and "NOT" not in result

        # Determine confidence based on grounding + relevance score
        relevance_score = state.get("relevance_score", 0.5)
        if is_grounded and relevance_score >= 0.7:
            confidence = "HIGH"
        elif is_grounded and relevance_score >= 0.3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        ms = int((time.time() - t0) * 1000)
        steps.append(
            f"[{ms}ms] Hallucination Check → {'GROUNDED' if is_grounded else 'NOT GROUNDED'} "
            f"(confidence: {confidence})"
        )
        timings.append({"node": "hallucination_check", "duration_ms": ms})

        # If not grounded, prepend a disclaimer
        final_answer = answer
        if not is_grounded:
            final_answer = (
                "**Note:** This answer may contain information not fully supported by the available sources. "
                "Please verify the details.\n\n" + answer
            )

        logger.info(f"Hallucination check: grounded={is_grounded}, confidence={confidence}")
        return {
            "answer": final_answer, "is_grounded": is_grounded, "confidence": confidence,
            "reasoning": " → ".join(steps),
            "reasoning_steps": steps, "node_timings": timings,
        }

    except Exception as e:
        logger.error(f"Hallucination check error: {e}")
        steps.append(f"Hallucination Check error → skipped")
        timings.append({"node": "hallucination_check", "duration_ms": 0})
        return {
            "is_grounded": True, "confidence": "MEDIUM",
            "reasoning": " → ".join(steps),
            "reasoning_steps": steps, "node_timings": timings,
        }


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def after_classify(state: AgentState) -> Literal["reformulate_query", "web_search", "generate_answer"]:
    """After classification, route to the right pipeline."""
    route = state.get("route", "DIRECT")
    if route == "RAG":
        return "reformulate_query"
    elif route == "WEB":
        return "web_search"
    return "generate_answer"


def after_grading(state: AgentState) -> Literal["generate_answer", "web_supplement_search", "web_fallback_search"]:
    """
    Three-way routing after grading:
      - HIGH relevance (>= 0.7): docs alone are sufficient → generate
      - PARTIAL (0.2 - 0.7): docs + web supplement → web_supplement_search
      - LOW (< 0.2): docs irrelevant → web_fallback_search (discard docs)
    """
    relevance = state.get("relevance_score", 0.0)
    has_web = state.get("has_web_search", False)

    if relevance >= 0.7:
        return "generate_answer"
    elif relevance >= 0.2 and has_web:
        return "web_supplement_search"
    elif has_web:
        return "web_fallback_search"
    return "generate_answer"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

def build_agent_graph():
    """
    Build the Corrective RAG state graph (8 nodes, 3 conditional edges).

    Graph:
      START → classify_intent
        ├─ DIRECT → generate_answer → hallucination_check → END
        ├─ WEB → web_search → generate_answer → hallucination_check → END
        └─ RAG → reformulate_query → retrieve_documents → grade_documents
                    ├─ RELEVANT → generate_answer → hallucination_check → END
                    └─ NOT RELEVANT → web_fallback_search → generate_answer → hallucination_check → END
    """
    graph = StateGraph(AgentState)

    # --- 9 Nodes ---
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("reformulate_query", reformulate_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("web_supplement_search", web_supplement_search)
    graph.add_node("web_fallback_search", web_fallback_search)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("hallucination_check", hallucination_check)

    # --- Entry ---
    graph.set_entry_point("classify_intent")

    # --- After classification: RAG → reformulate, WEB → search, DIRECT → generate ---
    graph.add_conditional_edges(
        "classify_intent", after_classify,
        {"reformulate_query": "reformulate_query", "web_search": "web_search", "generate_answer": "generate_answer"},
    )

    # --- RAG pipeline ---
    graph.add_edge("reformulate_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")

    # --- 3-way grading: HIGH → generate, PARTIAL → supplement + generate, LOW → fallback + generate ---
    graph.add_conditional_edges(
        "grade_documents", after_grading,
        {
            "generate_answer": "generate_answer",
            "web_supplement_search": "web_supplement_search",
            "web_fallback_search": "web_fallback_search",
        },
    )

    # --- Web supplement (partial docs + web) → generate ---
    graph.add_edge("web_supplement_search", "generate_answer")

    # --- Web fallback (docs irrelevant) → generate ---
    graph.add_edge("web_fallback_search", "generate_answer")

    # --- Web search (direct WEB route) → generate ---
    graph.add_edge("web_search", "generate_answer")

    # --- All generation → hallucination check → END ---
    graph.add_edge("generate_answer", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    return graph.compile()


# =============================================================================
# HELPERS
# =============================================================================

def _build_messages(response_mode, chat_history, user_content):
    """Build LangChain message list."""
    mode_prompt = CONCISE_PROMPT if response_mode == "concise" else DETAILED_PROMPT
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{mode_prompt}"
    messages = [SystemMessage(content=system_prompt)]
    if chat_history:
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_content))
    return messages


def _do_web_search(query, tavily_key=None):
    """Multi-tier web search: Tavily → Google Search + scraping."""
    from utils.web_search import web_search as unified_web_search
    return unified_web_search(query, tavily_key)


# =============================================================================
# COMPILED GRAPH (singleton)
# =============================================================================

_compiled_graph = build_agent_graph()


# =============================================================================
# PUBLIC API
# =============================================================================

def run_agent_graph(query, chat_history, llm, router_llm, retriever,
                    response_mode="concise", has_web_search=False, tavily_key=None):
    """Run the Corrective RAG graph (non-streaming)."""
    state = _build_initial_state(query, chat_history, llm, router_llm,
                                  retriever, response_mode, has_web_search, tavily_key)
    result = _compiled_graph.invoke(state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "route": result["route"],
        "reasoning": result["reasoning"],
        "confidence": result.get("confidence", "MEDIUM"),
        "node_timings": result.get("node_timings", []),
        "reasoning_steps": result.get("reasoning_steps", []),
    }


def run_agent_graph_stream(query, chat_history, llm, router_llm, retriever,
                           response_mode="concise", has_web_search=False, tavily_key=None):
    """
    Streaming wrapper. Yields UI-compatible dicts:
      - {"type": "thinking", "step": str}   (live node execution trace)
      - {"type": "route", "route": str}
      - {"type": "token", "token": str}
      - {"type": "complete", ...}
    """
    state = _build_initial_state(query, chat_history, llm, router_llm,
                                  retriever, response_mode, has_web_search, tavily_key)

    try:
        # Run full graph
        result = _compiled_graph.invoke(state)
        route = result.get("route", "DIRECT")
        steps = result.get("reasoning_steps", [])
        confidence = result.get("confidence", "MEDIUM")

        # Yield each reasoning step as a "thinking" event
        for step in steps:
            yield {"type": "thinking", "step": step}

        # Yield route badge
        yield {"type": "route", "route": route}

        # Stream answer
        answer = result.get("answer", "")
        chunk_size = 4
        for i in range(0, len(answer), chunk_size):
            yield {"type": "token", "token": answer[i:i + chunk_size]}

        yield {
            "type": "complete",
            "answer": answer,
            "sources": result.get("sources", []),
            "route": route,
            "reasoning": result.get("reasoning", ""),
            "confidence": confidence,
            "node_timings": result.get("node_timings", []),
            "reasoning_steps": steps,
        }

    except Exception as e:
        logger.error(f"Graph stream error: {e}")
        yield {
            "type": "complete",
            "answer": f"An error occurred: {str(e)}",
            "sources": [], "route": "ERROR",
            "reasoning": f"Error: {str(e)}", "confidence": "LOW",
        }


def _build_initial_state(query, chat_history, llm, router_llm,
                          retriever, response_mode, has_web_search, tavily_key):
    """Build the initial state dict for graph invocation."""
    return {
        "query": query,
        "reformulated_query": "",
        "chat_history": chat_history or [],
        "response_mode": response_mode,
        "route": "",
        "retrieved_docs": [],
        "doc_context": "",
        "doc_sources": [],
        "doc_sources_str": "",
        "docs_relevant": False,
        "relevance_score": 0.0,
        "web_results": [],
        "web_context": "",
        "answer": "",
        "sources": [],
        "is_grounded": True,
        "confidence": "MEDIUM",
        "reasoning": "",
        "reasoning_steps": [],
        "node_timings": [],
        "llm": llm,
        "router_llm": router_llm,
        "retriever": retriever,
        "has_web_search": has_web_search,
        "tavily_key": tavily_key,
    }
