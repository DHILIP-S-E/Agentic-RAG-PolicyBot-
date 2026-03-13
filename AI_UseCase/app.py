"""
PolicyBot — Corporate Policy Intelligence Assistant
Main Streamlit Application with Corrective RAG (CRAG) Agent Graph
"""

import streamlit as st
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from config.config import (
    get_default_provider, get_default_model, get_default_temperature,
    get_api_key, SUPPORTED_FILE_TYPES,
)
from models.llm import get_llm, get_router_llm
from models.embeddings import get_embedding_model
from utils.document_loader import load_uploaded_documents, load_documents_from_directory, chunk_documents
from utils.vector_store import create_vector_store, get_retriever, add_documents_to_store
from utils.website_loader import load_single_page, load_website
from utils.agent_graph import run_agent_graph_stream
from ui.components import (
    inject_css, render_route_badge, render_confidence_badge,
    render_source_cards, render_reasoning_trace,
    render_knowledge_status, COMMON_POLICY_QUESTIONS,
)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with all configuration options."""
    with st.sidebar:
        st.title("🏢 PolicyBot")
        st.caption("Corporate Policy Intelligence Assistant")

        # Navigation
        page = st.radio("Navigate", ["Chat", "Instructions"], index=0, label_visibility="collapsed")

        st.divider()

        # --- LLM Configuration ---
        st.subheader("🤖 LLM Settings")

        provider = get_default_provider()
        model_name = get_default_model()
        api_key_input = get_api_key(provider)
        temperature = get_default_temperature()

        provider_labels = {"groq": "Groq", "xai": "xAI Grok", "openai": "OpenAI", "gemini": "Google Gemini"}
        st.info(f"**Provider:** {provider_labels.get(provider, provider)}\n\n**Model:** `{model_name}`")

        if api_key_input:
            st.success("API Key: Configured")
        else:
            st.error(f"Set `{provider.upper()}_API_KEY` in your `.env` file")

        # --- Response Mode ---
        st.divider()
        st.subheader("📝 Response Mode")
        response_mode = st.radio(
            "Mode",
            ["Concise", "Detailed"],
            index=0,
            horizontal=True,
            help="Concise: Short 2-3 sentence answers. Detailed: Comprehensive explanations.",
        )

        # --- Knowledge Base ---
        st.divider()
        st.subheader("📄 Knowledge Base")

        # Upload documents
        uploaded_files = st.file_uploader(
            "Upload Policy Documents",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            help="Upload your company's HR policies, compliance manuals, handbooks (PDF/DOCX/TXT).",
        )

        # Company website URL
        st.markdown("**🌐 Add Company Website / Policy URL**")
        url_input = st.text_input(
            "Website URL",
            placeholder="https://yourcompany.com/policies",
            help="Paste a URL to scrape policy content. Supports HTML pages and PDF links.",
            label_visibility="collapsed",
        )

        col_url1, col_url2 = st.columns(2)
        with col_url1:
            add_single_url = st.button("🔗 Add Page", use_container_width=True,
                                       help="Scrape a single page")
        with col_url2:
            crawl_website = st.button("🕸️ Crawl Site", use_container_width=True,
                                      help="Crawl multiple pages from this URL (max 10)")

        # Knowledge base status
        doc_count = st.session_state.get("doc_count", 0)
        chunk_count = st.session_state.get("chunk_count", 0)
        scraped_urls = st.session_state.get("scraped_urls", [])
        uploaded_names = st.session_state.get("uploaded_file_names", set())

        render_knowledge_status(doc_count, chunk_count, scraped_urls, uploaded_names)

        # --- Knowledge Sources Toggle ---
        st.divider()
        st.subheader("🔍 Knowledge Sources")

        use_documents = st.checkbox("📄 Documents", value=True, help="Search uploaded documents")
        use_web_search = st.checkbox("🌐 Web Search", value=True, help="Fall back to web search if needed")

        # --- Vector Database ---
        st.divider()
        qdrant_url = os.environ.get("QDRANT_URL", "")
        qdrant_api_key = get_api_key("qdrant") or ""

        if qdrant_url:
            st.caption("🗄️ Qdrant: Connected")
        else:
            st.caption("🗄️ ChromaDB (in-memory)")

        tavily_key = get_api_key("tavily") or ""
        if tavily_key:
            st.caption("🔎 Tavily: Configured")
        else:
            st.caption("🔎 Web Search + Scraping (free)")

        # --- Actions ---
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            clear_btn = st.button("🗑️ Clear Chat", use_container_width=True)
        with col2:
            export_btn = st.button("📥 Export", use_container_width=True)

        if clear_btn:
            st.session_state.messages = []
            st.session_state.message_metadata = []
            st.rerun()

        if export_btn:
            _export_chat()

        msg_count = len(st.session_state.get("messages", []))
        if msg_count > 0:
            st.caption(f"💬 {msg_count} messages this session")

    return {
        "page": page,
        "provider": provider,
        "model_name": model_name,
        "api_key": api_key_input,
        "temperature": temperature,
        "response_mode": response_mode.lower(),
        "uploaded_files": uploaded_files,
        "url_input": url_input,
        "add_single_url": add_single_url,
        "crawl_website": crawl_website,
        "qdrant_url": qdrant_url,
        "qdrant_api_key": qdrant_api_key,
        "tavily_key": tavily_key,
        "use_documents": use_documents,
        "use_web_search": use_web_search,
    }


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def process_documents(config):
    """Process uploaded documents and URLs into the vector store."""
    try:
        embedding_model = get_embedding_model()
        new_documents = []

        # Process uploaded files
        uploaded_files = config["uploaded_files"]
        if uploaded_files:
            current_file_names = {f.name for f in uploaded_files}
            prev_file_names = st.session_state.get("uploaded_file_names", set())

            if current_file_names != prev_file_names:
                with st.spinner("📄 Loading and chunking documents..."):
                    docs = load_uploaded_documents(uploaded_files)
                    new_documents.extend(docs)
                    st.session_state.uploaded_file_names = current_file_names
                    st.toast(f"Loaded {len(docs)} document(s)", icon="📄")

        # Process single URL
        url_input = config.get("url_input", "")
        if config.get("add_single_url") and url_input and url_input.strip():
            url_clean = url_input.strip()
            scraped_urls = st.session_state.get("scraped_urls", [])
            already = any(item["url"] == url_clean for item in scraped_urls)

            if already:
                st.toast(f"URL already added", icon="ℹ️")
            else:
                with st.spinner(f"🔗 Scraping {url_clean}..."):
                    doc = load_single_page(url_clean)
                    if doc:
                        new_documents.append(doc)
                        if "scraped_urls" not in st.session_state:
                            st.session_state.scraped_urls = []
                        st.session_state.scraped_urls.append({
                            "url": url_clean,
                            "title": doc.metadata.get("source", url_clean),
                            "chars": doc.metadata.get("char_count", len(doc.page_content)),
                            "type": doc.metadata.get("type", "website"),
                        })
                        st.toast(f"Added: {doc.metadata.get('source', url_clean)}", icon="✅")
                    else:
                        st.error(
                            f"Could not extract content from: {url_clean}\n\n"
                            "The site may block scraping, require JavaScript, or the URL may be invalid."
                        )

        # Process website crawl
        if config.get("crawl_website") and url_input and url_input.strip():
            url_clean = url_input.strip()
            with st.spinner(f"🕸️ Crawling {url_clean} (up to 10 pages)..."):
                docs = load_website(url_clean, max_pages=10)
                if docs:
                    new_documents.extend(docs)
                    if "scraped_urls" not in st.session_state:
                        st.session_state.scraped_urls = []
                    for doc in docs:
                        doc_url = doc.metadata.get("url", url_clean)
                        if not any(item["url"] == doc_url for item in st.session_state.scraped_urls):
                            st.session_state.scraped_urls.append({
                                "url": doc_url,
                                "title": doc.metadata.get("source", doc_url),
                                "chars": doc.metadata.get("char_count", len(doc.page_content)),
                                "type": doc.metadata.get("type", "website"),
                            })
                    st.toast(f"Crawled {len(docs)} pages from {url_clean}", icon="🕸️")
                else:
                    st.error(f"Could not crawl any pages from: {url_clean}")

        # Load pre-seeded documents on first run
        if "preseeded_loaded" not in st.session_state:
            data_dir = os.path.join(os.path.dirname(__file__), "data", "documents")
            if os.path.exists(data_dir) and os.listdir(data_dir):
                with st.spinner("📚 Loading sample policy documents..."):
                    preseeded_docs = load_documents_from_directory(data_dir)
                    new_documents.extend(preseeded_docs)
            st.session_state.preseeded_loaded = True

        # Connect to Qdrant if configured
        qdrant_url = config.get("qdrant_url", "")
        qdrant_api_key = config.get("qdrant_api_key")
        if qdrant_url and "vector_store" not in st.session_state:
            with st.spinner("🗄️ Connecting to Qdrant..."):
                st.session_state.vector_store = create_vector_store(
                    [], embedding_model,
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key or None,
                )
                st.session_state.doc_count = st.session_state.get("doc_count", 0)
                st.session_state.chunk_count = st.session_state.get("chunk_count", 0)

        # Chunk and index new documents
        if new_documents:
            with st.spinner("🔍 Indexing into knowledge base..."):
                chunks = chunk_documents(new_documents)

                if "vector_store" in st.session_state and st.session_state.vector_store is not None:
                    add_documents_to_store(st.session_state.vector_store, chunks)
                else:
                    st.session_state.vector_store = create_vector_store(
                        chunks, embedding_model,
                        qdrant_url=qdrant_url or None,
                        qdrant_api_key=qdrant_api_key or None,
                    )

                st.session_state.doc_count = st.session_state.get("doc_count", 0) + len(new_documents)
                st.session_state.chunk_count = st.session_state.get("chunk_count", 0) + len(chunks)

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")


# =============================================================================
# CHAT PAGE
# =============================================================================

def chat_page(config):
    """Main chat interface."""
    # Header
    st.title("🏢 PolicyBot")
    st.caption("Corporate Policy Intelligence Assistant — Ask about your company's policies, compliance, and workplace procedures")

    # Validate API key
    if not config["api_key"]:
        st.warning(
            "⚠️ Please configure your API key in the `.env` file to start chatting.\n\n"
            "Check the **Instructions** page for setup help."
        )
        return

    # Process documents
    process_documents(config)

    # Initialize LLMs
    try:
        llm = get_llm(config["provider"], config["model_name"], config["api_key"], config["temperature"])

        groq_key = get_api_key("groq")
        if groq_key:
            try:
                router_llm = get_router_llm(groq_key)
            except Exception:
                router_llm = get_llm(config["provider"], config["model_name"], config["api_key"], 0.0)
        else:
            router_llm = get_llm(config["provider"], config["model_name"], config["api_key"], 0.0)

    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return

    # Get retriever
    retriever = None
    if config["use_documents"] and "vector_store" in st.session_state and st.session_state.vector_store is not None:
        retriever = get_retriever(st.session_state.vector_store)

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "message_metadata" not in st.session_state:
        st.session_state.message_metadata = []

    # Show quick action buttons when no messages yet
    if not st.session_state.messages:
        _render_welcome()

    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and i < len(st.session_state.message_metadata):
                meta = st.session_state.message_metadata[i]
                render_route_badge(meta.get("route", "DIRECT"))

            st.markdown(message["content"])

            if message["role"] == "assistant" and i < len(st.session_state.message_metadata):
                meta = st.session_state.message_metadata[i]
                if meta.get("confidence") and meta.get("route") != "DIRECT":
                    render_confidence_badge(meta["confidence"])
                render_source_cards(meta.get("sources", []))
                render_reasoning_trace(
                    meta.get("reasoning", ""), meta.get("route", ""),
                    meta.get("reasoning_steps"), meta.get("node_timings"),
                )

    # Check for suggestion clicks
    suggestion = st.session_state.pop("suggestion_click", None)

    # Chat input
    prompt = st.chat_input("Ask about your company policies, compliance, leave, benefits...")
    if suggestion:
        prompt = suggestion

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.message_metadata.append({})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            thinking_container = st.container()
            thinking_placeholder = thinking_container.empty()
            response_placeholder = st.empty()

            full_response = ""
            route = "DIRECT"
            sources = []
            reasoning = ""
            confidence = "MEDIUM"
            reasoning_steps = []
            node_timings = []
            thinking_steps = []

            try:
                stream = run_agent_graph_stream(
                    query=prompt,
                    chat_history=st.session_state.messages[:-1],
                    llm=llm,
                    router_llm=router_llm,
                    retriever=retriever,
                    response_mode=config["response_mode"],
                    has_web_search=config["use_web_search"],
                    tavily_key=config["tavily_key"] or None,
                )

                for item in stream:
                    if item.get("type") == "thinking":
                        thinking_steps.append(item["step"])
                        thinking_md = "**🔄 Analyzing your question...**\n\n"
                        for step in thinking_steps:
                            thinking_md += f"  `→` {step}\n\n"
                        thinking_placeholder.markdown(thinking_md)

                    elif item.get("type") == "route":
                        route = item["route"]
                        thinking_placeholder.empty()
                        render_route_badge(route)

                    elif item.get("type") == "token":
                        full_response += item["token"]
                        response_placeholder.markdown(full_response + "▌")

                    elif item.get("type") == "complete":
                        full_response = item.get("answer", full_response)
                        sources = item.get("sources", [])
                        reasoning = item.get("reasoning", "")
                        route = item.get("route", route)
                        confidence = item.get("confidence", "MEDIUM")
                        reasoning_steps = item.get("reasoning_steps", [])
                        node_timings = item.get("node_timings", [])

                # Final render
                response_placeholder.markdown(full_response)
                if route != "DIRECT":
                    render_confidence_badge(confidence)
                render_source_cards(sources)
                render_reasoning_trace(reasoning, route, reasoning_steps, node_timings)

            except Exception as e:
                thinking_placeholder.empty()
                full_response = f"I encountered an error: {str(e)}. Please try again."
                response_placeholder.markdown(full_response)

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.message_metadata.append({
                "route": route,
                "sources": sources,
                "reasoning": reasoning,
                "confidence": confidence,
                "reasoning_steps": reasoning_steps,
                "node_timings": node_timings,
            })


def _render_welcome():
    """Render welcome message with quick action buttons."""
    st.markdown(
        """
        Welcome! I can help you understand your company's policies and compliance requirements.

        **Get started:**
        - 📄 **Upload** your company's policy documents in the sidebar
        - 🌐 **Add** your company's policy website URL
        - 💬 **Ask** any question about workplace policies
        """
    )

    # Quick action buttons
    st.markdown("**Quick Questions:**")
    cols = st.columns(3)
    questions = COMMON_POLICY_QUESTIONS
    for i, q in enumerate(questions):
        with cols[i % 3]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state["suggestion_click"] = q
                st.rerun()


# =============================================================================
# INSTRUCTIONS PAGE
# =============================================================================

def instructions_page():
    """Instructions and setup guide."""
    st.title("🏢 PolicyBot — Setup Guide")
    st.caption("Corporate Policy Intelligence Assistant")

    st.markdown("""
## What is PolicyBot?

PolicyBot is an **AI-powered policy assistant** that helps employees instantly understand their company's
policies by reading official documents and company websites.

**Instead of reading 100-page policy documents, employees just ask PolicyBot.**

---

## Who is it for?

| Role | Use Case |
|------|----------|
| **Employees** | "How many sick leaves do I have?" "Can I work remotely?" |
| **HR Teams** | Quick policy reference, onboarding support |
| **Compliance Teams** | Regulatory compliance questions |
| **Managers** | Policy clarification for team decisions |

---

## How It Works

PolicyBot answers from **three knowledge sources:**
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
### 📄 Source 1: Documents
Upload your company's policy files:
- HR Policy PDFs
- Employee Handbook
- Compliance Manuals
- Data Privacy Guidelines

*Uses RAG (Retrieval-Augmented Generation)*
""")
    with col2:
        st.markdown("""
### 🌐 Source 2: Company Website
Paste your company's policy URL:
- Scrapes page content
- Crawls linked pages
- Supports PDF links
- Converts to searchable knowledge

*Automatic text extraction*
""")
    with col3:
        st.markdown("""
### 🔍 Source 3: Web Search
If info isn't in your documents:
- Searches the web automatically
- Finds relevant regulations
- GDPR, labor laws, etc.
- Falls back intelligently

*Tavily or Web Search + Scraping*
""")

    st.markdown("---")

    # Agent Graph
    st.subheader("🔄 Intelligent Answer Flow (Corrective RAG)")
    st.markdown("PolicyBot uses a **LangGraph StateGraph** — not simple function calls:")

    st.code("""
    User Question
          │
          ▼
    ┌─────────────┐
    │ 1. Classify  │──── DIRECT ───► Generate Answer
    │    Intent    │
    └──────┬──────┘
           │
      RAG  │  WEB
           │    └──────► Web Search ──► Generate Answer
           ▼
    ┌─────────────┐
    │ 2. Reformulate│   (handles follow-up questions)
    │    Query     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 3. Retrieve  │   (from your documents)
    │  Documents   │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 4. Grade     │   (are docs actually relevant?)
    │  Documents   │
    └───┬─────┬───┘
        │     │
     YES│     │NO → Web Fallback Search
        ▼     ▼
    ┌─────────────┐
    │ 5. Generate  │   (using all context)
    │   Answer     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 6. Hallucin. │   (is answer grounded?)
    │   Check      │
    └──────┬──────┘
           ▼
        Response + Confidence Score
    """, language=None)

    st.markdown("""
**Key Features:**
- **Corrective RAG** — If retrieved documents aren't relevant, automatically falls back to web search
- **Query Reformulation** — Follow-up questions are rewritten using chat context
- **Hallucination Guard** — Every answer is checked for factual grounding
- **Confidence Scoring** — 🟢 HIGH / 🟡 MEDIUM / 🔴 LOW

---

## Quick Start

### Step 1: Configure API Key
Set your LLM provider in the `.env` file:

| Provider | Free? | Speed |
|----------|-------|-------|
| **Groq** | Yes | Very Fast |
| **Google Gemini** | Yes (free tier) | Fast |
| **OpenAI** | Paid | Fast |

### Step 2: Add Your Company's Knowledge
- **Upload files**: PDF, DOCX, or TXT in the sidebar
- **Paste URL**: Company website or policy page link
- **Crawl site**: Automatically scrape multiple pages

### Step 3: Ask Questions
Navigate to **Chat** and ask anything about your company's policies!

---

## Example Questions

| Question | Source Used |
|----------|-----------|
| "What is our maternity leave policy?" | 📄 Documents |
| "How does our remote work policy compare to industry standards?" | 📄 + 🌐 Corrective RAG |
| "What are the latest GDPR requirements?" | 🔍 Web Search |
| "Summarize our employee handbook" | 📄 Documents |
| "What is the reimbursement process?" | 📄 Documents |

---

Ready to start? Navigate to the **Chat** page!
    """)


# =============================================================================
# CHAT EXPORT
# =============================================================================

def _export_chat():
    """Export chat history as Markdown download."""
    if "messages" not in st.session_state or not st.session_state.messages:
        st.sidebar.warning("No chat history to export.")
        return

    markdown = f"# PolicyBot Chat Export\n"
    markdown += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"

    for i, msg in enumerate(st.session_state.messages):
        role = "**You**" if msg["role"] == "user" else "**PolicyBot**"

        if msg["role"] == "assistant" and i < len(st.session_state.get("message_metadata", [])):
            meta = st.session_state.message_metadata[i]
            route = meta.get("route", "")
            if route:
                role += f" [{route}]"

        markdown += f"{role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        "📥 Download Chat",
        markdown,
        file_name=f"policybot_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title="PolicyBot — Corporate Policy Intelligence",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    inject_css()

    config = render_sidebar()

    if config["page"] == "Instructions":
        instructions_page()
    else:
        chat_page(config)


if __name__ == "__main__":
    main()
