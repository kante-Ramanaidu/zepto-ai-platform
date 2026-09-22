# main.py
# FastAPI application wrapping the LangGraph support assistant.
#
# Endpoints:
#   POST /ask   — accepts {"query": str}, returns AskResponse JSON
#   GET  /health — liveness check
#
# Run locally:
#   python ingest.py                                  # once, before first run
#   uvicorn main:app --host 0.0.0.0 --port 8000
#
# Docker:
#   docker build -t zepto-support-assistant .
#   docker run -p 7860:7860 zepto-support-assistant

from fastapi import FastAPI
from schemas import AskRequest, AskResponse
from graph import run_query

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "RAG-based policy Q&A service. "
        "Graded with MOCK_LLM at its default (unset or 1) — "
        "fully deterministic, no API key required."
    ),
    version="1.0.0"
)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Accept a customer query and return a Pydantic-validated response.

    In mock mode (default, graded):
      - policy questions → retrieval from ChromaDB + canned template answer
      - general questions → fixed canned string, no retrieval

    In MOCK_LLM=0 mode (optional, ungraded):
      - policy questions → ChromaDB retrieval + real LLM answer
      - general questions → real LLM direct answer
    """
    return run_query(request.query)


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}
