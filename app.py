"""
Streamlit Web UI for Industry-Style Agentic RAG.
Features:
- Chat interface with full message history
- Real-time agent execution trace (Routing, Retrieval, Grading, Web Fallback, Rewrite)
- Source attribution (PDF paper filename, page numbers, Web URLs)
- Knowledge base management & live PDF ingestion trigger
- Sidebar status check for API keys & model settings
"""

import os
import glob
import time
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Agentic RAG | Federated Learning & IDS Knowledge Base",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-kb {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #a5d6a7;
    }
    .badge-web {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #90caf9;
    }
    .badge-direct {
        background-color: #f3e5f5;
        color: #6a1b9a;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #ce93d8;
    }
    .badge-insufficient {
        background-color: #fff3e0;
        color: #e65100;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #ffcc80;
    }
    .metric-card {
        padding: 12px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imports from project modules
from config import (
    GROQ_API_KEY,
    TAVILY_API_KEY,
    PINECONE_API_KEY,
    HF_TOKEN,
    GROQ_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PDF_DIR,
)
from graph import ask_agent, get_agent_app, AgentState
from config import get_embeddings
from database import (
    init_db,
    save_message,
    load_chat_messages,
    load_recent_history_for_llm,
    clear_chat_history,
    count_chat_messages,
)


@st.cache_resource(show_spinner="Loading embeddings model and initializing agent...")
def load_cached_agent():
    """Initializes and caches the embeddings model and LangGraph workflow in memory once."""
    get_embeddings()
    return get_agent_app()


# Initialize SQLite Database on startup
init_db()


# =========================================================================
# Sidebar: System Configuration & Status
# =========================================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain_stack.png",
        width=180,
    )
    st.title("🛡️ Agentic RAG Control")
    st.markdown("---")

    # API Keys Health Check
    st.subheader("🔑 API Key Status")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Groq LLM:")
        st.write("Pinecone:")
        st.write("Tavily Search:")
        st.write("HF Token:")
    with col2:
        st.markdown("🟢 `Active`" if GROQ_API_KEY else "🔴 `Missing`")
        st.markdown("🟢 `Active`" if PINECONE_API_KEY else "🔴 `Missing`")
        st.markdown("🟢 `Active`" if TAVILY_API_KEY else "🔴 `Missing`")
        st.markdown("🟢 `Active`" if HF_TOKEN else "🟡 `Optional`")

    st.markdown("---")

    # Knowledge Base Stats
    st.subheader("📚 PDF Knowledge Base")
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    st.write(f"📁 Folder: `{PDF_DIR}/`")
    st.write(f"📄 Papers Loaded: **{len(pdf_files)} PDFs**")
    st.write(f"🗄️ Index: `{PINECONE_INDEX_NAME}`")
    st.write(f"🏷️ Namespace: `{PINECONE_NAMESPACE}`")

    # Ingestion Button
    if st.button("⚡ Ingest / Re-index PDFs", use_container_width=True):
        if not PINECONE_API_KEY:
            st.error("PINECONE_API_KEY is missing in .env!")
        else:
            with st.spinner("Indexing PDF documents into Pinecone... This may take a few moments."):
                try:
                    from ingest import run_ingestion
                    run_ingestion()
                    st.success("✅ PDF documents successfully indexed into Pinecone!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.markdown("---")

    # Quick Sample Questions
    st.subheader("💡 Sample Questions")
    sample_queries = [
        "How does federated learning detect network intrusions?",
        "What are the defense mechanisms against adversarial or collusion attacks in FL?",
        "What is Tavily Search and why is it useful for AI agents?",
        "Hello! What topics are in your private knowledge base?",
    ]

    for q in sample_queries:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["pending_query"] = q

    st.markdown("---")

    # SQLite Database Management
    st.markdown("---")
    st.subheader("💾 Chat History Database")
    msg_count = count_chat_messages()
    st.write("🗄️ Database: `db.sqlite3`")
    st.write(f"💬 Messages Stored: **{msg_count}**")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I help you today?",
                "source": "direct",
                "trace": None,
            }
        ]
        st.rerun()

    st.markdown("---")

    # Workflow Architecture Expander
    with st.expander("🔍 Workflow Architecture"):
        st.markdown(
            """
            **Decision Pipeline:**
            1. **Router**: Dispatches to Private KB or Direct Chat.
            2. **Pinecone Vector Search**: Retrieves top semantic chunks from 47 PDF research papers.
            3. **KB Grader**: LLM verifies if retrieved chunks answer the question (`good` vs `weak`).
            4. **Tavily Fallback**: If KB is weak, queries the live web.
            5. **Web Grader**: Verifies relevance of web snippets.
            6. **Query Rewriter**: Reformulates query on weak evidence with loop guard.
            7. **Grounded Generation**: Produces cited response with source badge.
            """
        )


# =========================================================================
# Main Content
# =========================================================================
st.markdown('<div class="main-header">Industry-Style Agentic RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Autonomous Fallback Architecture powered by <b>LangGraph</b>, <b>Groq</b>, <b>Pinecone</b> & <b>Tavily</b></div>',
    unsafe_allow_html=True,
)

# Initialize Session State Messages from db.sqlite3
if "messages" not in st.session_state:
    stored_messages = load_chat_messages()
    if stored_messages:
        st.session_state.messages = stored_messages
    else:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your **Agentic Cybersecurity & Research Assistant**. "
                    "I have access to your private repository of **47 research papers on Federated Learning, Intrusion Detection Systems (IDS), and Network Security**. "
                    "Our chat history is persistently stored in **`db.sqlite3`** and I maintain conversational memory of our prior turns. "
                    "\n\nHow can I help you today?"
                ),
                "source": "direct",
                "trace": None,
            }
        ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Source badge for assistant responses
        if msg["role"] == "assistant" and msg.get("source"):
            source_type = msg["source"]
            if source_type in ("private_kb", "kb"):
                st.markdown('<span class="badge-kb">🛡️ Source: Private PDF Research KB</span>', unsafe_allow_html=True)
            elif source_type in ("web_search", "web"):
                st.markdown('<span class="badge-web">🌐 Source: Tavily Web Search Fallback</span>', unsafe_allow_html=True)
            elif source_type == "direct":
                st.markdown('<span class="badge-direct">💬 Source: Direct Response</span>', unsafe_allow_html=True)
            elif source_type == "insufficient_evidence":
                st.markdown('<span class="badge-insufficient">⚠️ Source: Insufficient Evidence</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        # Display trace if available
        if msg.get("trace"):
            with st.expander("🔬 Agent Decision & Evidence Trace"):
                trace = msg["trace"]
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.write(f"**Router Decision:** `{trace.get('source_used')}`")
                    st.write(f"**KB Evidence Grade:** `{trace.get('kb_grade', 'N/A')}`")
                with t_col2:
                    st.write(f"**Web Evidence Grade:** `{trace.get('web_grade', 'N/A')}`")
                    st.write(f"**Retries / Rewrites:** `{trace.get('retry_count', 0)}`")

                if trace.get("current_query") and trace.get("current_query") != msg.get("question"):
                    st.write(f"**Rewritten Query:** `{trace.get('current_query')}`")

                # Show retrieved KB chunks
                kb_docs = trace.get("kb_docs", [])
                if kb_docs:
                    st.markdown("---")
                    st.markdown(f"**Retrieved Private KB Chunks ({len(kb_docs)}):**")
                    for idx, doc in enumerate(kb_docs, 1):
                        fname = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "PDF"))
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {idx}:** `{fname}` *(Page {page})*")
                        st.caption(doc.page_content[:400] + "...")

                # Show Web Search Results if used
                web_res = trace.get("web_results", "")
                if web_res:
                    st.markdown("---")
                    st.markdown("**Tavily Web Search Results:**")
                    st.caption(web_res[:800] + ("..." if len(web_res) > 800 else ""))


# =========================================================================
# Handle User Input
# =========================================================================
user_prompt = st.chat_input("Ask a question about Federated Learning, Intrusion Detection, or any topic...")

# Check if a sample query was clicked
if "pending_query" in st.session_state and st.session_state["pending_query"]:
    user_prompt = st.session_state.pop("pending_query")

if user_prompt:
    # 1. Save and display User message
    save_message(role="user", content=user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Retrieve recent dialogue history for conversational context
    recent_history = load_recent_history_for_llm(max_turns=6)

    # 3. Execute Agentic RAG
    with st.chat_message("assistant"):
        try:
            app = load_cached_agent()
            initial_state: AgentState = {
                "question": user_prompt,
                "current_query": user_prompt,
                "chat_history": recent_history,
                "kb_docs": [],
                "web_results": "",
                "kb_grade": "",
                "web_grade": "",
                "answer": "",
                "source_used": "",
                "retry_count": 0,
            }

            accumulated_state = dict(initial_state)

            # Live Step-by-Step Reasoning Progress
            with st.status("🧠 Agent Reasoning in progress...", expanded=True) as status_box:
                for step_update in app.stream(initial_state, stream_mode="updates"):
                    for node_name, node_output in step_update.items():
                        accumulated_state.update(node_output)

                        if node_name == "route_question":
                            route = node_output.get("source_used", "")
                            status_box.write(f"🚦 **Question Router:** Dispatched query to `{route}`")
                        elif node_name == "retrieve_kb":
                            docs = node_output.get("kb_docs", [])
                            status_box.write(f"📚 **Pinecone Vector Search:** Retrieved `{len(docs)}` chunks from PDF papers")
                        elif node_name == "grade_kb_evidence":
                            grade = node_output.get("kb_grade", "")
                            status_box.write(f"⚖️ **Evidence Grader:** Assessed private KB relevance → `{grade}`")
                        elif node_name == "search_web":
                            status_box.write("🌐 **Tavily Fallback:** KB evidence was insufficient, querying live web...")
                        elif node_name == "grade_web_evidence":
                            grade = node_output.get("web_grade", "")
                            status_box.write(f"🌐 **Web Grader:** Evaluated web search relevance → `{grade}`")
                        elif node_name == "rewrite_query":
                            rewritten = node_output.get("current_query", "")
                            status_box.write(f"🔄 **Query Rewriter:** Reformulated query → *'{rewritten}'*")
                        elif node_name in ("generate_from_kb", "generate_from_web", "direct_answer", "answer_insufficient"):
                            status_box.write(f"✍️ **Answer Generator:** Composing final response from `{node_name}`...")

                status_box.update(label="✅ Reasoning complete!", state="complete", expanded=False)

            source_type = accumulated_state.get("source_used", "direct")
            answer_text = accumulated_state.get("answer", "No response generated.")

            # Badge display
            if source_type in ("private_kb", "kb"):
                st.markdown('<span class="badge-kb">🛡️ Source: Private PDF Research KB</span>', unsafe_allow_html=True)
            elif source_type in ("web_search", "web"):
                st.markdown('<span class="badge-web">🌐 Source: Tavily Web Search Fallback</span>', unsafe_allow_html=True)
            elif source_type == "direct":
                st.markdown('<span class="badge-direct">💬 Source: Direct Response</span>', unsafe_allow_html=True)
            elif source_type == "insufficient_evidence":
                st.markdown('<span class="badge-insufficient">⚠️ Source: Insufficient Evidence</span>', unsafe_allow_html=True)

            # Stream generated answer text smoothly
            def stream_text_chunks(text: str):
                import time
                words = text.split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                    time.sleep(0.012)

            st.write_stream(stream_text_chunks(answer_text))

            # Render Live Trace Expander
            with st.expander("🔬 Agent Decision & Evidence Trace"):
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.write(f"**Router Decision:** `{accumulated_state.get('source_used')}`")
                    st.write(f"**KB Evidence Grade:** `{accumulated_state.get('kb_grade', 'N/A')}`")
                with t_col2:
                    st.write(f"**Web Evidence Grade:** `{accumulated_state.get('web_grade', 'N/A')}`")
                    st.write(f"**Retries / Rewrites:** `{accumulated_state.get('retry_count', 0)}`")

                if accumulated_state.get("current_query") and accumulated_state.get("current_query") != user_prompt:
                    st.write(f"**Rewritten Query:** `{accumulated_state.get('current_query')}`")

                kb_docs = accumulated_state.get("kb_docs", [])
                if kb_docs:
                    st.markdown("---")
                    st.markdown(f"**Retrieved Private KB Chunks ({len(kb_docs)}):**")
                    for idx, doc in enumerate(kb_docs, 1):
                        fname = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "PDF"))
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {idx}:** `{fname}` *(Page {page})*")
                        st.caption(doc.page_content[:400] + "...")

                web_res = accumulated_state.get("web_results", "")
                if web_res:
                    st.markdown("---")
                    st.markdown("**Tavily Web Search Results:**")
                    st.caption(web_res[:800] + ("..." if len(web_res) > 800 else ""))

            # Save to SQLite db.sqlite3 database
            save_message(
                role="assistant",
                content=answer_text,
                source=source_type,
                trace=accumulated_state,
            )

            # Save to UI session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "source": source_type,
                    "question": user_prompt,
                    "trace": accumulated_state,
                }
            )

        except Exception as e:
            st.error(f"Error executing agent workflow: {e}")

