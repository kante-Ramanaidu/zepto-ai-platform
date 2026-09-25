# Module 3 — Support Assistant

## Overview

A complete GenAI support service for Zepto. Eight policy documents are embedded locally using `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in ChromaDB. A LangGraph `StateGraph` routes each query through a 3-node pipeline (classify → retrieve/answer → respond), and the result is served via a FastAPI `POST /ask` endpoint. The entire graded path is fully offline and deterministic — no API key, no network call to any LLM provider.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run

```bash
# Step 1: Ingest the 8 policy documents into ChromaDB (run once)
python ingest.py

# Step 2: Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker (alternative)
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# Endpoint: http://localhost:7860/ask
```

---

## MOCK_LLM Toggle

| Value | Behaviour | API key needed? | Graded? |
|-------|-----------|----------------|---------|
| Unset or `1` (default) | Deterministic mock — keyword heuristic + canned answers | No | **Yes — this is the graded path** |
| `0` | Routes to Groq free-tier LLM | Yes (`GROQ_API_KEY`) | No (optional extension) |

All example calls below and all recorded responses are with `MOCK_LLM` at its default.

---

## Example Calls (MOCK_LLM at default — graded)

### Call 1 — Policy question (triggers retrieval)

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the delivery fee for orders below INR 149?\"}"
```

**Raw JSON response (recorded from live server, MOCK_LLM at default):**
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}
```

- Query contains keyword `"delivery"` → routed to `policy_question` → `retrieve_and_answer`
- Top retrieved source: `doc_01` (Delivery Policy) — correct match for a delivery fee question
- Answer follows the required canned template: `"Based on the retrieved context: {top_chunk[:200]}"`
- Sources populated with the top-3 retrieved chunk IDs

### Call 2 — General question (no retrieval)

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the capital of France?\"}"
```

**Raw JSON response (recorded from live server, MOCK_LLM at default):**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

- Query contains no policy keywords → routed to `general_question` → `direct_answer`
- Fixed canned string returned — no retrieval, no LLM call
- `sources` is empty list as required

---

## RAG Pipeline Architecture

```
docs/doc_0X.txt  -->  [ingest.py]  -->  ChromaDB collection 'zepto_policies'
                                                  |
User query  -->  POST /ask  -->  [graph.py: classify_intent]
                                        |                   |
                               policy_question       general_question
                                        |                   |
                         [retrieve_and_answer]        [direct_answer]
                          ChromaDB top-3 query         canned string
                          + canned template            (mock mode)
                                        |
                              AskResponse (Pydantic)
                              {answer, sources, confidence}
```

### Stage-by-stage description

**Stage 1 — Ingestion** (`ingest.py`):
All 8 policy documents in `docs/` are read from disk. Each file is treated as one chunk (per-document chunking — sufficient given the short length of each policy paragraph). `ingest.py` is the only file that reads the raw `.txt` sources. Run once before starting the server.

**Stage 2 — Embedding** (`ingest.py`, model `all-MiniLM-L6-v2`):
Each document chunk is embedded using `sentence-transformers` locally. The model produces 384-dimensional dense vectors. No network call is made — the model runs entirely on-device. Embeddings are L2-normalised before storage to support cosine similarity.

**Stage 3 — Storage** (`ingest.py`, ChromaDB):
Embeddings, document texts, and metadata (source filename, doc_id) are stored in a persistent ChromaDB collection named `zepto_policies` at `./chroma_db`. The collection is configured with `hnsw:space = cosine`.

**Stage 4 — Intent Classification** (`graph.py` → `classify_intent` node):
Every query enters the graph at `classify_intent`. In mock mode (default, graded), a keyword heuristic checks whether the lowercased query contains any of: `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`. If yes → `policy_question`; otherwise → `general_question`. No LLM call is made in mock mode.

**Stage 5 — Routing** (`graph.py`, conditional edge):
A `conditional_edge` from `classify_intent` routes to `retrieve_and_answer` for policy questions or `direct_answer` for general questions. This routing logic does not depend on `MOCK_LLM` — only the generation steps do.

**Stage 6 — Retrieval** (`graph.py` → `retrieve_and_answer` node):
For policy questions, the query is embedded with the same `all-MiniLM-L6-v2` model and the top-3 most similar chunks are retrieved from ChromaDB via cosine similarity. This step **always runs for real in both mock and real-LLM modes** — it requires no API key and makes no external network call.

**Stage 7 — Generation** (`graph.py` → `retrieve_and_answer` / `direct_answer` nodes):
This is the only stage that branches on `MOCK_LLM`.
- **Mock mode (default, graded)**: `retrieve_and_answer` returns `f"Based on the retrieved context: {top_chunk[:200]}"` — deterministic, no LLM call. `direct_answer` returns the fixed canned string.
- **MOCK_LLM=0 mode (optional)**: `retrieve_and_answer` sends the structured prompt from `prompt.py` to Groq's LLM with the retrieved context. Retries up to 2 additional times if the response fails Pydantic validation. `direct_answer` calls the LLM directly without retrieval.

**Stage 8 — Structured Output** (`schemas.py`, `AskResponse`):
Every response — mock or real — is validated against the `AskResponse` Pydantic model before FastAPI serialises it to JSON. Fields: `answer` (str), `sources` (List[str]), `confidence` (float 0–1).

### MOCK_LLM toggle summary

| Stage | MOCK_LLM unset / 1 (graded) | MOCK_LLM=0 (optional) |
|-------|-----------------------------|-----------------------|
| Intent classification | Keyword heuristic (no LLM) | Groq LLM call |
| Retrieval | Always real (ChromaDB) | Always real (ChromaDB) |
| Answer — policy question | Canned template from top chunk | Groq LLM + structured prompt |
| Answer — general question | Fixed canned string | Groq LLM direct |

---

## Prompt Template (for MOCK_LLM=0 extension)

See `prompt.py` for the full text. The template follows the **role–context–task–format–length** skeleton and includes:
- One explicit negative constraint: `"Do NOT answer using information not present in the provided context."`
- One few-shot example embedded in the prompt body (delivery fee query + correct answer).

---

## Document Corpus

| File | Topic |
|------|-------|
| `docs/doc_01.txt` | Delivery Policy |
| `docs/doc_02.txt` | Returns & Refunds |
| `docs/doc_03.txt` | Membership Tiers |
| `docs/doc_04.txt` | Order Tracking |
| `docs/doc_05.txt` | Order Cancellation Policy |
| `docs/doc_06.txt` | Damaged or Missing Items |
| `docs/doc_07.txt` | Gift Cards |
| `docs/doc_08.txt` | Customer Support Hours |

---

## Docker

The Dockerfile pre-ingests all 8 documents at build time (`RUN python ingest.py`) so the container starts with ChromaDB already populated — no startup delay.

```bash
# Build
docker build -t zepto-support-assistant .

# Run (MOCK_LLM at default — no API key needed)
docker run -p 7860:7860 zepto-support-assistant

# Test
curl -X POST "http://localhost:7860/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the return policy?\"}"
```
