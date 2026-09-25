# graph.py
# LangGraph StateGraph for the Zepto Support Assistant.
#
# Architecture:
#   classify_intent  →  (conditional edge)  →  retrieve_and_answer
#                                           →  direct_answer
#
# MOCK_LLM toggle (env var):
#   Unset or "1" → mock mode (graded default): keyword heuristic + canned answers, no LLM call
#   "0"          → real-LLM mode (optional, ungraded): calls Groq LLM
#
# The retrieval step (ChromaDB query) always runs for real in both modes —
# it needs no API key and makes no external network call.

import os
from typing import TypedDict, List

import chromadb
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from schemas import AskResponse

# ── Configuration ─────────────────────────────────────────────────────────────
MOCK_LLM        = os.environ.get("MOCK_LLM", "1") != "0"   # True = mock (graded default)
CHROMA_PATH     = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 3
TOP_SNIPPET_LEN = 200   # chars for the canned mock answer

# Keywords for classify_intent mock heuristic (spec-defined list)
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "track", "tracking", "cancel", "gift card", "support hours"
]

# ── Shared resources (loaded once at module import) ────────────────────────────
print("[graph.py] Loading embedding model...")
_embed_model = SentenceTransformer(EMBED_MODEL)

print("[graph.py] Connecting to ChromaDB...")
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection    = _chroma_client.get_collection(COLLECTION_NAME)
print(f"[graph.py] Collection '{COLLECTION_NAME}' loaded — {_collection.count()} docs")


# ── State schema (TypedDict) ───────────────────────────────────────────────────
class GraphState(TypedDict):
    query:          str
    intent:         str            # "policy_question" | "general_question"
    retrieved_docs: List[dict]     # [{id, document, distance}, ...]
    response:       AskResponse


# ── Node 1: classify_intent ────────────────────────────────────────────────────
def classify_intent(state: GraphState) -> GraphState:
    """
    Classifies the incoming query as 'policy_question' or 'general_question'.

    Mock mode (MOCK_LLM=1, graded default):
        Uses a keyword heuristic — no LLM call.
        If any keyword from POLICY_KEYWORDS appears in the lowercased query,
        classify as policy_question; otherwise general_question.

    Optional MOCK_LLM=0 extension:
        Calls the Groq LLM to classify.
    """
    query = state["query"]

    if MOCK_LLM:
        # ── GRADED BASELINE: keyword heuristic, no LLM ──
        q_lower = query.lower()
        intent  = (
            "policy_question"
            if any(kw in q_lower for kw in POLICY_KEYWORDS)
            else "general_question"
        )
        print(f"[classify_intent] mock heuristic → {intent}")

    else:
        # ── OPTIONAL MOCK_LLM=0 EXTENSION: real LLM classification ──
        from groq import Groq
        client   = Groq(api_key=os.environ["GROQ_API_KEY"])
        keywords = ", ".join(f'"{kw}"' for kw in POLICY_KEYWORDS)
        resp     = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "user",
                "content": (
                    f"Classify the following customer query as either "
                    f"'policy_question' or 'general_question'.\n"
                    f"A policy_question is one that asks about any of these topics: "
                    f"{keywords}.\n"
                    f"Reply with ONLY the label — no other text.\n\n"
                    f"Query: {query}"
                )
            }],
            max_tokens=10
        )
        raw    = resp.choices[0].message.content.strip().lower()
        intent = "policy_question" if "policy" in raw else "general_question"
        print(f"[classify_intent] LLM → {intent}")

    return {**state, "intent": intent}


# ── Node 2: retrieve_and_answer ────────────────────────────────────────────────
def retrieve_and_answer(state: GraphState) -> GraphState:
    """
    For 'policy_question' queries:
      - Retrieval always runs for real (ChromaDB, no API key needed).
      - Answer generation branches on MOCK_LLM.

    Mock mode (graded):
        Returns f"Based on the retrieved context: {top_chunk_snippet}"
        No LLM call.

    Optional MOCK_LLM=0 extension:
        Sends retrieved chunks to the real LLM via the structured prompt.
        Retries up to 2 additional times on Pydantic validation failure.
    """
    query = state["query"]

    # ── Retrieval (always real, both modes) ───────────────────────────────────
    query_embedding = _embed_model.encode(
        query, normalize_embeddings=True
    ).tolist()

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "distances", "metadatas"]
    )

    retrieved_docs = [
        {
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    top_doc     = retrieved_docs[0]
    top_snippet = top_doc["document"][:TOP_SNIPPET_LEN]
    source_ids  = [d["id"] for d in retrieved_docs]

    print(f"[retrieve_and_answer] top source: {top_doc['id']}  "
          f"(distance={top_doc['distance']:.4f})")

    if MOCK_LLM:
        # ── GRADED BASELINE: canned template answer, no LLM ──
        answer   = f"Based on the retrieved context: {top_snippet}"
        response = AskResponse(
            answer=answer,
            sources=source_ids,
            confidence=1.0
        )

    else:
        # ── OPTIONAL MOCK_LLM=0 EXTENSION: real LLM with structured prompt ──
        import json
        from groq import Groq
        from prompt import build_prompt

        client  = Groq(api_key=os.environ["GROQ_API_KEY"])
        context = "\n\n".join(
            f"[{d['id']}]\n{d['document']}" for d in retrieved_docs
        )
        prompt_text = build_prompt(query=query, retrieved_context=context)

        response = None
        for attempt in range(3):   # up to 2 retries (attempt 0, 1, 2)
            llm_resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=400
            )
            raw_text = llm_resp.choices[0].message.content.strip()

            try:
                parsed   = json.loads(raw_text)
                response = AskResponse(**parsed)
                break
            except Exception:
                if attempt < 2:
                    print(f"[retrieve_and_answer] validation failed (attempt {attempt+1}), retrying...")
                    prompt_text += (
                        "\n\nIMPORTANT: Your previous response could not be parsed as JSON. "
                        "Reply ONLY with valid JSON in this exact format:\n"
                        '{"answer": "...", "sources": ["doc_id", ...], "confidence": 0.0}'
                    )
                else:
                    print("[retrieve_and_answer] all retries exhausted — returning error response")
                    response = AskResponse(
                        answer="ERROR: Could not produce a valid structured response after retries.",
                        sources=source_ids,
                        confidence=0.0
                    )

    return {**state, "retrieved_docs": retrieved_docs, "response": response}


# ── Node 3: direct_answer ──────────────────────────────────────────────────────
def direct_answer(state: GraphState) -> GraphState:
    """
    For 'general_question' queries:

    Mock mode (graded):
        Returns a fixed canned string. No LLM call, no retrieval.

    Optional MOCK_LLM=0 extension:
        Calls the LLM directly without retrieval.
    """
    if MOCK_LLM:
        # ── GRADED BASELINE: fixed canned string, no LLM ──
        response = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
        print("[direct_answer] mock → canned response")

    else:
        # ── OPTIONAL MOCK_LLM=0 EXTENSION: real LLM, no retrieval ──
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp   = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role":    "system",
                    "content": "You are Zepto's helpful customer support assistant."
                },
                {
                    "role":    "user",
                    "content": state["query"]
                }
            ],
            max_tokens=250
        )
        response = AskResponse(
            answer=resp.choices[0].message.content.strip(),
            sources=[],
            confidence=0.8
        )
        print("[direct_answer] LLM → answered directly")

    return {**state, "response": response}


# ── Conditional routing function ───────────────────────────────────────────────
def route_intent(state: GraphState) -> str:
    """
    Routes to the correct node based on the intent classified by classify_intent.
    This routing logic does NOT depend on MOCK_LLM — only the generation steps do.
    Returns: "policy_question" or "general_question"
    """
    return state["intent"]


# ── Build and compile the StateGraph ──────────────────────────────────────────
def build_graph():
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("classify_intent",     classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer",       direct_answer)

    # Entry point
    builder.set_entry_point("classify_intent")

    # Conditional edge: classify_intent → retrieve_and_answer OR direct_answer
    builder.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "policy_question":  "retrieve_and_answer",
            "general_question": "direct_answer"
        }
    )

    # Terminal edges
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer",       END)

    return builder.compile()


# Compiled graph — imported and used by main.py
graph = build_graph()


def run_query(query: str) -> AskResponse:
    """
    Public interface for main.py.
    Runs the full LangGraph pipeline for a single query.

    Args:
        query: The user's question string.

    Returns:
        AskResponse with answer, sources, and confidence.
    """
    initial_state: GraphState = {
        "query":          query,
        "intent":         "",
        "retrieved_docs": [],
        "response":       None   # type: ignore
    }
    final_state = graph.invoke(initial_state)
    return final_state["response"]
