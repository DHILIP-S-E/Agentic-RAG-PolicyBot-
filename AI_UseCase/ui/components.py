"""
UI Components — Professional Streamlit widgets for PolicyBot.
Source cards, confidence badges, suggestion chips, custom CSS.
"""

import streamlit as st


# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
<style>
/* Chat container styling */
.stChatMessage {
    border-radius: 12px;
}

/* Route badge */
.route-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 16px;
    font-size: 0.8em;
    font-weight: 600;
    color: white;
    margin-bottom: 8px;
}

/* Confidence badge */
.confidence-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: 500;
    color: white;
    margin-left: 8px;
}

/* Source card */
.source-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    transition: background 0.2s;
}
.source-card:hover {
    background: #e8f0fe;
}
.source-card .source-title {
    font-weight: 600;
    font-size: 0.9em;
    color: #1a73e8;
}
.source-card .source-meta {
    font-size: 0.78em;
    color: #5f6368;
    margin-top: 2px;
}
.source-card .source-excerpt {
    font-size: 0.82em;
    color: #3c4043;
    margin-top: 6px;
    line-height: 1.4;
}

/* Suggestion chips */
.suggestion-chip {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    background: #e8f0fe;
    color: #1a73e8;
    font-size: 0.85em;
    margin: 4px 4px;
    cursor: pointer;
    border: 1px solid #c2d9f7;
    transition: background 0.2s;
}
.suggestion-chip:hover {
    background: #c2d9f7;
}

/* Quick action buttons */
.quick-action {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    font-size: 0.85em;
    font-weight: 500;
    border: none;
    cursor: pointer;
    margin: 4px;
}

/* Knowledge source indicator */
.knowledge-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75em;
    font-weight: 500;
    margin: 2px;
}

/* Stats card */
.stats-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stats-card .stats-number {
    font-size: 1.8em;
    font-weight: 700;
    color: #1a73e8;
}
.stats-card .stats-label {
    font-size: 0.85em;
    color: #5f6368;
}
</style>
"""


def inject_css():
    """Inject custom CSS into the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# ROUTE BADGES
# =============================================================================

ROUTE_CONFIG = {
    "RAG": ("📚", "Document Search", "#1E88E5"),
    "RAG+WEB": ("🔄", "Corrective RAG + Web", "#7B1FA2"),
    "WEB": ("🌐", "Web Search", "#43A047"),
    "DIRECT": ("💬", "Direct Answer", "#757575"),
    "ERROR": ("⚠️", "Error", "#E53935"),
}


def render_route_badge(route):
    """Render a colored route indicator badge."""
    emoji, label, color = ROUTE_CONFIG.get(route, ("❓", "Unknown", "#999"))
    st.markdown(
        f'<span class="route-badge" style="background-color:{color};">'
        f'{emoji} {label}</span>',
        unsafe_allow_html=True,
    )


def render_confidence_badge(confidence):
    """Render confidence level badge."""
    conf_map = {
        "HIGH": ("🟢", "#2E7D32"),
        "MEDIUM": ("🟡", "#F9A825"),
        "LOW": ("🔴", "#C62828"),
    }
    icon, color = conf_map.get(confidence, ("⚪", "#999"))
    st.markdown(
        f'<span class="confidence-badge" style="background-color:{color};">'
        f'{icon} {confidence} confidence</span>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SOURCE CARDS
# =============================================================================

def render_source_cards(sources):
    """Render source citations as styled cards."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources):
            if "filename" in source:
                # Document source
                title = source.get("filename", "Unknown")
                meta = f"Page {source.get('page', 'N/A')}"
                excerpt = source.get("excerpt", "")
                icon = "📄"
            elif "url" in source:
                # Web source
                title = source.get("title", "Web Source")
                meta = source.get("url", "")
                excerpt = source.get("snippet", "")[:200]
                icon = "🌐"
            else:
                continue

            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-title">{icon} {title}</div>'
                f'<div class="source-meta">{meta}</div>'
                f'<div class="source-excerpt">{excerpt[:200]}{"..." if len(excerpt) > 200 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# REASONING TRACE
# =============================================================================

def render_reasoning_trace(reasoning, route, reasoning_steps=None, node_timings=None):
    """Render agent reasoning trace (collapsible)."""
    if not reasoning and not reasoning_steps:
        return

    with st.expander("🧠 Agent Reasoning Trace", expanded=False):
        st.markdown(f"**Route:** {route}")

        if reasoning_steps:
            st.markdown("**Execution Path:**")
            for step in reasoning_steps:
                st.markdown(f"  `→` {step}")

        if node_timings:
            total_ms = sum(t.get("duration_ms", 0) for t in node_timings)
            st.markdown(f"\n**Total processing time:** {total_ms}ms")

        if reasoning and not reasoning_steps:
            st.markdown(f"**Reasoning:** {reasoning}")


# =============================================================================
# QUESTION SUGGESTIONS
# =============================================================================

COMMON_POLICY_QUESTIONS = [
    "What is the leave policy?",
    "Remote work guidelines?",
    "How does reimbursement work?",
    "What is the code of conduct?",
    "Data privacy policy?",
    "What are employee benefits?",
]


def render_suggestions(on_click_key="suggestion_click"):
    """Render clickable question suggestion chips."""
    st.markdown("**💡 Common Questions:**")
    cols = st.columns(3)
    for i, q in enumerate(COMMON_POLICY_QUESTIONS):
        with cols[i % 3]:
            if st.button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state[on_click_key] = q


def render_follow_up_suggestions(suggestions):
    """Render follow-up question suggestions after a response."""
    if not suggestions:
        return
    st.markdown("**💡 You may also ask:**")
    cols = st.columns(min(len(suggestions), 3))
    for i, q in enumerate(suggestions):
        with cols[i % len(cols)]:
            if st.button(q, key=f"followup_{i}", use_container_width=True):
                st.session_state["suggestion_click"] = q


# =============================================================================
# KNOWLEDGE BASE STATUS
# =============================================================================

def render_knowledge_status(doc_count, chunk_count, scraped_urls, uploaded_names):
    """Render knowledge base status with stats."""
    if doc_count == 0 and not scraped_urls:
        st.info("📭 No knowledge base yet. Upload documents or add URLs above to get started.")
        return

    # Stats row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stats-card">'
            f'<div class="stats-number">{doc_count}</div>'
            f'<div class="stats-label">Sources</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stats-card">'
            f'<div class="stats-number">{chunk_count}</div>'
            f'<div class="stats-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    # Uploaded files
    if uploaded_names:
        with st.expander(f"📄 Documents ({len(uploaded_names)})", expanded=False):
            for name in uploaded_names:
                st.markdown(f"- 📄 {name}")

    # Scraped URLs
    if scraped_urls:
        with st.expander(f"🌐 Websites ({len(scraped_urls)})", expanded=False):
            for item in scraped_urls:
                st.markdown(f"- 🔗 **{item['title']}** ({item['chars']:,} chars)\n  `{item['url']}`")
