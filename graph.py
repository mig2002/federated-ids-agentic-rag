"""
LangGraph Workflow Definition for Industry-Style Agentic RAG.
Features:
- Routing (Private KB vs Direct Answer)
- Vector Retrieval from Private Pinecone KB
- LLM-based Grading of Retrieved Evidence
- Fallback to Web Search via Tavily when KB is weak
- Grading of Web Evidence
- Query Rewriting with Retry Guard
- Grounded Generation with Source Tracking
"""

import os
from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from config import get_llm, get_retriever, get_tavily_search


# =========================================================================
# Structured Decision Schemas
# =========================================================================
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Use kb for questions needing Agentic RAG docs; direct for greetings/simple chat."
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="good means evidence can answer the question; weak means not enough evidence."
    )


class AgentState(TypedDict):
    question: str
    current_query: str
    chat_history: List[dict]
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int


def format_history_context(chat_history: List[dict], max_turns: int = 4) -> str:
    """Formats recent chat history for prompt context."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history[-max_turns:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\nPrevious Conversation:\n" + "\n".join(lines) + "\n"


# =========================================================================
# Node 1 — Route the Question (with Conversation Memory)
# =========================================================================
def route_question(state: AgentState):
    question = state["question"]
    chat_history = state.get("chat_history", [])
    llm = get_llm()
    router_llm = llm.with_structured_output(RouteDecision, method="json_mode")

    history_str = format_history_context(chat_history)

    prompt = f"""You are a router for an Agentic RAG assistant with a private knowledge base of research papers on Federated Learning (FL), Intrusion Detection Systems (IDS), Network Security, IoT/Edge Computing, and Blockchain.
{history_str}
Route to "kb" if the user asks about (or refers in follow-up to):
- Federated learning, split learning, personalized/horizontal FL
- Intrusion detection systems (IDS/NIDS), anomaly detection, cyber attack detection
- Machine learning models for IoT, IIoT, vehicular networks (VANET/IoV)
- Adversarial attacks, defense mechanisms, collusion, privacy preservation in FL
- Explainable AI (FedXAI) and robustness in security models
- Specific papers, methods, or algorithms in the research database
- Follow-up questions about any of the above topics from previous conversation

Route to "direct" only for pure greetings, thanks, or very simple chat.

Current Question:
{question}

Return your response as valid JSON.
Example:
{{"route": "kb"}}
"""
    decision = router_llm.invoke(prompt)
    print(f"[Router] {decision.route}")

    # If follow-up with chat history, contextualize the search query for Pinecone
    current_query = question
    if chat_history and decision.route == "kb":
        try:
            standalone_prompt = f"""Given the following previous conversation and a follow-up question, rephrase the follow-up question into a standalone, specific search query for document retrieval.
Do not answer the question, just return the search query.

{history_str}
Follow-up Question: {question}

Standalone Query:"""
            standalone_query = llm.invoke(standalone_prompt).content.strip()
            if standalone_query:
                current_query = standalone_query
                print(f"[Contextualized Query] '{question}' -> '{current_query}'")
        except Exception:
            current_query = question

    return {
        "current_query": current_query,
        "source_used": decision.route,
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"


# =========================================================================
# Node 2 — Retrieve from Private KB
# =========================================================================
def retrieve_kb(state: AgentState):
    query = state["current_query"]
    retriever = get_retriever()
    docs = retriever.invoke(query)

    print(f"[KB Retriever] Query: {query}")
    print(f"[KB Retriever] Retrieved: {len(docs)} chunks")

    return {"kb_docs": docs}


# =========================================================================
# Node 3 — Grade Private KB Evidence
# =========================================================================
def grade_kb_evidence(state: AgentState):
    question = state["question"]
    llm = get_llm()
    kb_grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

    context_chunks = []
    for doc in state.get("kb_docs", []):
        fname = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "PDF"))
        page = doc.metadata.get("page")
        page_str = f" (Page {page})" if page else ""
        context_chunks.append(f"Source: {fname}{page_str}\n{doc.page_content}")
    context = "\n\n".join(context_chunks)

    prompt = f"""You are an evidence grader.

Question:
{question}

Private KB evidence:
{context}

Can this private KB evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
"""
    grade = kb_grader_llm.invoke(prompt)
    print(f"[KB Grader] {grade.grade}")

    return {"kb_grade": grade.grade}


def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"


# =========================================================================
# Node 4 — Search Web (Tavily Fallback)
# =========================================================================
def search_web(state: AgentState):
    query = state["current_query"]
    print(f"[Tavily Search] Query: {query}")

    web_search = get_tavily_search()
    result = web_search.invoke({"query": query})

    if isinstance(result, dict):
        answer = result.get("answer", "")
        results = result.get("results", [])
        lines = []
        if answer:
            lines.append(f"Tavily answer: {answer}")

        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")

        web_text = "\n\n".join(lines) if lines else str(result)
    else:
        web_text = str(result)

    print(f"[Tavily Search] Result characters: {len(web_text)}")

    return {
        "web_results": web_text,
        "source_used": "web",
    }


# =========================================================================
# Node 5 — Grade Web Evidence
# =========================================================================
def grade_web_evidence(state: AgentState):
    question = state["question"]
    web_results = state.get("web_results", "")
    llm = get_llm()
    web_grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

    prompt = f"""You are an evidence grader.

Question:
{question}

Web search evidence:
{web_results}

Can this web evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
"""
    grade = web_grader_llm.invoke(prompt)
    print(f"[Web Grader] {grade.grade}")

    return {"web_grade": grade.grade}


MAX_RETRIES = 1


def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite_query"

    return "answer_insufficient"


# =========================================================================
# Node 6 — Rewrite the Query
# =========================================================================
def rewrite_query(state: AgentState):
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    llm = get_llm()

    prompt = f"""Rewrite the question for better retrieval and web search.

Rules:
- Preserve original intent.
- Make it specific and search-friendly.
- Do not answer.
- Return only the rewritten query.

Original question:
{question}
"""
    rewritten = llm.invoke(prompt).content.strip()
    print(f"[Rewriter] {rewritten}")

    return {
        "current_query": rewritten,
        "retry_count": retry_count,
    }


# =========================================================================
# Node 7 — Generate from Private KB
# =========================================================================
def generate_from_kb(state: AgentState):
    question = state["question"]
    chat_history = state.get("chat_history", [])
    history_str = format_history_context(chat_history)
    llm = get_llm()

    context_chunks = []
    for doc in state.get("kb_docs", []):
        fname = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "PDF"))
        page = doc.metadata.get("page")
        page_str = f" (Page {page})" if page else ""
        context_chunks.append(f"[KB Source: {fname}{page_str}]\n{doc.page_content}")
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a technical research assistant and instructor.

Answer the question using the provided private PDF research paper context. Maintain conversational flow with previous discussion if applicable.

Rules:
- Provide an accurate, comprehensive, and well-structured answer.
- Do not invent unsupported details.
- Cite the relevant paper title/filename and page number(s) if present in the context.
- Mention that the answer is based on the private PDF knowledge base.
- Include source type: Private PDF KB.
{history_str}
Question:
{question}

Private KB context:
{context}
"""
    answer = llm.invoke(prompt).content

    return {
        "answer": answer,
        "source_used": "private_kb",
    }


# =========================================================================
# Node 8 — Generate from Web Search
# =========================================================================
def generate_from_web(state: AgentState):
    question = state["question"]
    chat_history = state.get("chat_history", [])
    history_str = format_history_context(chat_history)
    web_context = state.get("web_results", "")
    llm = get_llm()

    prompt = f"""You are a technical instructor.

The private KB was insufficient, so web search was used. Maintain conversational flow with previous discussion if applicable.

Answer using ONLY the web search context.

Rules:
- Beginner-friendly explanation.
- Do not invent unsupported details.
- Mention that the answer is based on Tavily web search.
- Include source type: Web Search.
- If URLs are present in context, include the most useful URLs.
{history_str}
Question:
{question}

Web search context:
{web_context}
"""
    answer = llm.invoke(prompt).content

    return {
        "answer": answer,
        "source_used": "web_search",
    }


# =========================================================================
# Node 9 — Direct Answer
# =========================================================================
def direct_answer(state: AgentState):
    question = state["question"]
    chat_history = state.get("chat_history", [])
    history_str = format_history_context(chat_history)
    llm = get_llm()

    prompt = f"""Respond naturally and conversationally.
{history_str}
Message:
{question}
"""
    answer = llm.invoke(prompt).content

    return {
        "answer": answer,
        "source_used": "direct",
    }


# =========================================================================
# Node 10 — Insufficient Evidence Answer
# =========================================================================
def answer_insufficient(state: AgentState):
    answer = (
        "I could not find enough reliable evidence in the private knowledge base "
        "or the web search results to answer this confidently. "
        "Please provide more specific documents or rephrase the question."
    )

    return {
        "answer": answer,
        "source_used": "insufficient_evidence",
    }


# =========================================================================
# LangGraph Workflow Assembly
# =========================================================================
def create_agent_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve_kb", retrieve_kb)
    workflow.add_node("grade_kb_evidence", grade_kb_evidence)
    workflow.add_node("search_web", search_web)
    workflow.add_node("grade_web_evidence", grade_web_evidence)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate_from_kb", generate_from_kb)
    workflow.add_node("generate_from_web", generate_from_web)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("answer_insufficient", answer_insufficient)

    # Add Edges & Conditional Edges
    workflow.add_edge(START, "route_question")

    workflow.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge("retrieve_kb", "grade_kb_evidence")

    workflow.add_conditional_edges(
        "grade_kb_evidence",
        decide_after_kb_grade,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    workflow.add_edge("search_web", "grade_web_evidence")

    workflow.add_conditional_edges(
        "grade_web_evidence",
        decide_after_web_grade,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve_kb")
    workflow.add_edge("generate_from_kb", END)
    workflow.add_edge("generate_from_web", END)
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("answer_insufficient", END)

    app = workflow.compile()
    return app


_compiled_app = None
def get_agent_app():
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = create_agent_graph()
    return _compiled_app


def ask_agent(question: str, chat_history: Optional[List[dict]] = None, app=None):
    """Executes the Agentic RAG workflow for a given question and displays results."""
    if app is None:
        app = get_agent_app()

    initial_state: AgentState = {
        "question": question,
        "current_query": question,
        "chat_history": chat_history or [],
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 90)
    print("QUESTION:")
    print(question)
    print("\nSOURCE USED:")
    print(result.get("source_used", "unknown"))
    print("\nFINAL ANSWER:")
    print(result.get("answer", ""))
    print("=" * 90)

    return result
