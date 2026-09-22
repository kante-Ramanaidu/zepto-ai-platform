# Module 3 — Support Assistant (`/support_assistant`) — 25 Marks

## Tech Stack
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) — local, no API key
- **Vector store**: `chromadb` — local, no API key
- **Orchestration**: `langgraph` + `langchain-core`
- **Structured output**: `pydantic` (v2)
- **API server**: `fastapi` + `uvicorn`
- **Containerization**: `Docker`
- **LLM (optional)**: `groq` — only when `MOCK_LLM=0` (ungraded extension)
- **Environment**: `python-dotenv`

> **KEY RULE**: `MOCK_LLM` unset or `=1` → fully deterministic mock (no network, no API key) — **this is what gets graded**. Only `MOCK_LLM=0` calls a real LLM.

---

## Folder Structure to Create

```
/support_assistant
├── docs/
│   ├── doc_01.txt    ← Delivery Policy
│   ├── doc_02.txt    ← Returns & Refunds
│   ├── doc_03.txt    ← Membership Tiers
│   ├── doc_04.txt    ← Order Tracking
│   ├── doc_05.txt    ← Order Cancellation Policy
│   ├── doc_06.txt    ← Damaged or Missing Items
│   ├── doc_07.txt    ← Gift Cards
│   └── doc_08.txt    ← Customer Support Hours
├── ingest.py         ← Load docs, embed with all-MiniLM-L6-v2, store in ChromaDB
├── graph.py          ← LangGraph StateGraph (3 nodes + conditional edge)
├── prompt.py         ← Structured prompt template
├── schemas.py        ← Pydantic models (request + response)
├── main.py           ← FastAPI app with POST /ask endpoint
├── Dockerfile        ← Container definition
├── requirements.txt  ← All dependencies
└── README.md         ← Architecture description + example call transcripts
```

---

## Step-by-Step Implementation

---

### STEP 1 — Create the 8 Document Files

Create `/support_assistant/docs/` and populate each file with the **exact text** from the project spec.

**`docs/doc_01.txt`**:
```
Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.
```

**`docs/doc_02.txt`**:
```
Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.
```

**`docs/doc_03.txt`**:
```
Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period.
```

**`docs/doc_04.txt`**:
```
Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.
```

**`docs/doc_05.txt`**:
```
Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.
```

**`docs/doc_06.txt`**:
```
If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed.
```

**`docs/doc_07.txt`**:
```
Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law.
```

**`docs/doc_08.txt`**:
```
Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered.
```

---

### STEP 2 — `requirements.txt`

```
sentence-transformers
chromadb
langgraph
langchain-core
pydantic
fastapi
uvicorn[standard]
python-dotenv
# Optional (only needed if MOCK_LLM=0):
# groq
```

---

### STEP 3 — `schemas.py` — Pydantic Models

```python
# schemas.py
from pydantic import BaseModel, Field
from typing import List

class AskRequest(BaseModel):
    query: str = Field(..., description="The user's question")

class AskResponse(BaseModel):
    answer:     str         = Field(..., description="The answer to the query")
    sources:    List[str]   = Field(default_factory=list,
                                    description="Chunk/document IDs used (empty for general questions)")
    confidence: float       = Field(..., ge=0.0, le=1.0,
                                    description="Confidence score between 0 and 1")
```

---

### STEP 4 — `ingest.py` — Embed All 8 Documents into ChromaDB

```python
# ingest.py
import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR    = "docs"
COLLECTION  = "zepto_policies"
MODEL_NAME  = "all-MiniLM-L6-v2"

def ingest():
    # Load embedding model (local, no API key)
    model = SentenceTransformer(MODEL_NAME)

    # Init ChromaDB (persistent local storage)
    client     = chromadb.PersistentClient(path="./chroma_db")

    # Delete and recreate collection for clean re-ingestion
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    doc_files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".txt"))

    documents, ids, embeddings, metadatas = [], [], [], []

    for doc_file in doc_files:
        doc_id   = doc_file.replace(".txt", "")   # e.g. "doc_01"
        filepath = os.path.join(DOCS_DIR, doc_file)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Simple per-document chunking (one chunk per doc — sufficient given their length)
        # For finer granularity you could split into ~200-char fixed-size chunks here
        embedding = model.encode(text).tolist()

        documents.append(text)
        ids.append(doc_id)
        embeddings.append(embedding)
        metadatas.append({"source": doc_file})

        print(f"  Ingested: {doc_id}")

    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"\n✅ Ingested {len(ids)} documents into ChromaDB collection '{COLLECTION}'")
    return collection

if __name__ == "__main__":
    ingest()
```

**Run once before starting the app**:
```bash
python ingest.py
```

---

### STEP 5 — `prompt.py` — Structured Prompt Template

This is used by the **optional** `MOCK_LLM=0` path. It must be present in code (graded for structure), even though mock mode never calls it.

```python
# prompt.py

# Structured prompt template following the role–context–task–format–length skeleton.
# Used only when MOCK_LLM=0 (optional real-LLM extension).
# Present in code for grading; never invoked in default mock mode.

SYSTEM_PROMPT = """
## Role
You are Zepto's AI support assistant. You answer customer questions strictly based on
Zepto's official policy documents provided to you as context. You are helpful, concise,
and accurate.

## Context
The following are excerpts from Zepto's official policy documents, retrieved specifically
for this query. Use ONLY this information to answer:

{retrieved_context}

## Task
Answer the customer's query accurately and completely using only the information in the
context above. If the answer is not present in the context, say so explicitly.

## Format
- Respond in clear, plain English.
- If multiple policies are relevant, address each briefly.
- Do NOT add information not present in the provided context.
- Do NOT speculate or make assumptions beyond what the documents state.

## Length
Keep your answer under 150 words unless the query genuinely requires more detail.

## Negative Constraint
Do NOT answer using information not present in the provided context. If the context does
not contain a clear answer, respond: "I don't have enough information in the current
policies to answer that. Please contact Zepto support directly."

## Few-Shot Example
Query: "How much does delivery cost for an order of INR 100?"
Context excerpt: "Standard delivery is free on orders over INR 149; orders below this
threshold incur a flat INR 25 delivery fee."
Answer: "For an order of INR 100, which is below the INR 149 free-delivery threshold,
a flat delivery fee of INR 25 applies."

## Customer Query
{query}
"""

def build_prompt(query: str, retrieved_context: str) -> str:
    return SYSTEM_PROMPT.format(query=query, retrieved_context=retrieved_context)
```

---

### STEP 6 — `graph.py` — LangGraph StateGraph (3 Nodes + Conditional Edge)

```python
# graph.py
import os
import chromadb
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from schemas import AskResponse

# ── Configuration ────────────────────────────────────────────────────────────
MOCK_LLM    = os.environ.get("MOCK_LLM", "1") != "0"   # True = mock (graded default)
COLLECTION  = "zepto_policies"
MODEL_NAME  = "all-MiniLM-L6-v2"
TOP_K       = 3

# ── Keyword list for intent classification ────────────────────────────────────
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]

# ── Shared resources (loaded once at module import) ───────────────────────────
_embed_model = SentenceTransformer(MODEL_NAME)
_chroma      = chromadb.PersistentClient(path="./chroma_db")
_collection  = _chroma.get_collection(COLLECTION)


# ── State schema ──────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    query:          str
    intent:         str           # "policy_question" | "general_question"
    retrieved_docs: List[dict]    # list of {id, document, distance}
    response:       AskResponse


# ── Node 1: classify_intent ───────────────────────────────────────────────────
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if MOCK_LLM:
        # Graded baseline: keyword heuristic — no LLM call
        q_lower = query.lower()
        intent  = "policy_question" if any(kw in q_lower for kw in POLICY_KEYWORDS) \
                  else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call real LLM to classify
        # (Groq or other free-tier LLM)
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp   = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "user",
                "content": (
                    f"Classify the following query as either 'policy_question' or "
                    f"'general_question'.\n"
                    f"A policy_question is about Zepto's delivery, returns, refunds, "
                    f"membership, tracking, cancellation, gift cards, or support hours.\n"
                    f"Reply with ONLY the label.\n\nQuery: {query}"
                )
            }],
            max_tokens=10
        )
        raw    = resp.choices[0].message.content.strip().lower()
        intent = "policy_question" if "policy" in raw else "general_question"

    return {**state, "intent": intent}


# ── Node 2: retrieve_and_answer ───────────────────────────────────────────────
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both modes (embedding + ChromaDB need no API key)
    query_embedding = _embed_model.encode(query).tolist()
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "distances", "metadatas"]
    )

    retrieved_docs = [
        {
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i]
        }
        for i in range(len(results["ids"][0]))
    ]

    top_doc     = retrieved_docs[0]
    top_snippet = top_doc["document"][:200]   # first ~200 chars
    source_ids  = [d["id"] for d in retrieved_docs]

    if MOCK_LLM:
        # Graded baseline: canned templated answer — no LLM call
        answer = f"Based on the retrieved context: {top_snippet}"
        response = AskResponse(answer=answer, sources=source_ids, confidence=1.0)
    else:
        # Optional MOCK_LLM=0 extension: call real LLM with structured prompt
        from groq import Groq
        from prompt import build_prompt
        import json

        client  = Groq(api_key=os.environ["GROQ_API_KEY"])
        context = "\n\n".join(d["document"] for d in retrieved_docs)
        prompt  = build_prompt(query=query, retrieved_context=context)

        raw_answer = None
        for attempt in range(3):   # retry up to 2 additional times on validation failure
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            raw_text = resp.choices[0].message.content.strip()
            try:
                # Try to parse as JSON first; fall back to plain text answer
                parsed = json.loads(raw_text)
                response = AskResponse(**parsed)
                break
            except Exception:
                if attempt < 2:
                    prompt += (
                        "\n\nIMPORTANT: Your previous response could not be parsed. "
                        "Reply ONLY with valid JSON matching: "
                        '{"answer": "...", "sources": [...], "confidence": 0.0-1.0}'
                    )
                else:
                    response = AskResponse(
                        answer="ERROR: Could not generate a valid structured response.",
                        sources=source_ids,
                        confidence=0.0
                    )

    return {**state, "retrieved_docs": retrieved_docs, "response": response}


# ── Node 3: direct_answer ─────────────────────────────────────────────────────
def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        # Graded baseline: fixed canned string — no LLM call
        answer   = "I can only answer questions about Zepto policies right now."
        response = AskResponse(answer=answer, sources=[], confidence=1.0)
    else:
        # Optional MOCK_LLM=0 extension: call real LLM directly (no retrieval)
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp   = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role":    "system",
                "content": "You are Zepto's helpful support assistant."
            }, {
                "role":    "user",
                "content": state["query"]
            }],
            max_tokens=200
        )
        answer   = resp.choices[0].message.content.strip()
        response = AskResponse(answer=answer, sources=[], confidence=0.8)

    return {**state, "response": response}


# ── Conditional routing ───────────────────────────────────────────────────────
def route_intent(state: GraphState) -> str:
    # Routing does NOT depend on MOCK_LLM — only generation steps do
    return state["intent"]   # "policy_question" | "general_question"


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classify_intent",    classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer",      direct_answer)

    builder.set_entry_point("classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "policy_question":  "retrieve_and_answer",
            "general_question": "direct_answer"
        }
    )

    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer",       END)

    return builder.compile()


# Compiled graph — imported by main.py
graph = build_graph()


def run_query(query: str) -> AskResponse:
    """Public interface — call this from FastAPI."""
    initial_state: GraphState = {
        "query":          query,
        "intent":         "",
        "retrieved_docs": [],
        "response":       None
    }
    final_state = graph.invoke(initial_state)
    return final_state["response"]
```

---

### STEP 7 — `main.py` — FastAPI App with POST `/ask`

```python
# main.py
from fastapi import FastAPI
from schemas import AskRequest, AskResponse
from graph import run_query

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based policy Q&A — graded with MOCK_LLM at default (mock mode)",
    version="1.0.0"
)

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_query(request.query)

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Run locally**:
```bash
# First, ingest documents (run once):
python ingest.py

# Then start the server:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### STEP 8 — Test the App & Record Example Calls for README

Run these in a terminal (or in a notebook cell) while the server is running. Record the **raw JSON responses** in the README.

**Example Call 1 — Policy question (should trigger retrieval)**:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the delivery fee for orders below INR 149?\"}"
```
Expected response shape:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials...",
  "sources": ["doc_01", "doc_02", "doc_04"],
  "confidence": 1.0
}
```

**Example Call 2 — General question (should NOT trigger retrieval)**:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the capital of France?\"}"
```
Expected response shape:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

**Verify routing in Python**:
```python
from graph import run_query

# Policy question
r1 = run_query("How do I cancel my order?")
print(r1)   # sources should be non-empty, answer should reference retrieved context

# General question
r2 = run_query("Tell me a joke")
print(r2)   # sources = [], canned answer
```

---

### STEP 9 — `Dockerfile`

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Pre-ingest documents at build time so the container starts ready
RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Build and run locally**:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# App available at http://localhost:7860
```

**Document in README**: these exact `docker build` + `docker run` commands. This locally-runnable Dockerfile is what gets graded.

---

### STEP 10 — Write `support_assistant/README.md`

Must contain:

#### 1. Install & Run Steps
```bash
cd support_assistant
pip install -r requirements.txt
python ingest.py          # embed + load all 8 docs into ChromaDB
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 2. Example Call Transcripts (MOCK_LLM at default)
Paste the raw JSON responses from both curl calls above.

#### 3. RAG Pipeline Architecture Description
Write these stages in order (required by grader):

**Stage 1 — Ingestion** (`ingest.py`): Eight policy documents in `docs/` are read from disk. Each document is treated as a single chunk (simple per-document chunking). No external API is needed.

**Stage 2 — Embedding** (`ingest.py`, `all-MiniLM-L6-v2`): Each chunk is embedded locally using `sentence-transformers` with the `all-MiniLM-L6-v2` model. Embeddings are 384-dimensional vectors. This runs entirely on-device with no network call.

**Stage 3 — Storage** (`ingest.py`, ChromaDB collection `zepto_policies`): Embeddings, document texts, and IDs are stored in a persistent ChromaDB collection at `./chroma_db`. The collection is queryable by cosine similarity.

**Stage 4 — Retrieval** (`graph.py → retrieve_and_answer node`): At query time, the user's query is embedded with the same `all-MiniLM-L6-v2` model. ChromaDB returns the top-3 most similar chunks by cosine similarity. This step runs for real in **both mock and real-LLM modes** — it requires no API key.

**Stage 5 — Intent Classification** (`graph.py → classify_intent node`): Before retrieval, the query is routed. In mock mode (default, graded): a keyword heuristic checks for policy-related terms. In real-LLM mode (`MOCK_LLM=0`): an LLM classifies the intent instead. The conditional edge then routes to `retrieve_and_answer` or `direct_answer`.

**Stage 6 — Generation** (`graph.py → retrieve_and_answer / direct_answer nodes`): This is the stage that branches on `MOCK_LLM`. In mock mode (default, graded): `retrieve_and_answer` returns `"Based on the retrieved context: {top_snippet}"` — no network call. `direct_answer` returns a fixed canned string. In real-LLM mode (`MOCK_LLM=0`): the structured prompt from `prompt.py` is sent to a Groq LLM with the retrieved context, and the response is validated against the Pydantic schema with up to 2 retries.

**Stage 7 — Structured Output** (`schemas.py`): Every response — mock or real — is validated and serialized as a `AskResponse(answer, sources, confidence)` Pydantic model before being returned by FastAPI.

**MOCK_LLM toggle summary**:
| Stage | MOCK_LLM=1 (default, graded) | MOCK_LLM=0 (optional) |
|---|---|---|
| Intent classification | Keyword heuristic | LLM call |
| Retrieval | Always real (ChromaDB) | Always real (ChromaDB) |
| Answer generation (policy) | Canned template from top chunk | LLM with structured prompt |
| Answer generation (general) | Fixed canned string | LLM direct call |

#### 4. Docker Build + Run Instructions
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# Test: curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the return policy?"}'
```

---

## Acceptance Criteria Checklist (Self-check before submitting)

- [ ] All 8 `docs/doc_0X.txt` files present with exact text from the spec
- [ ] `python ingest.py` runs without error and loads all 8 docs into ChromaDB
- [ ] Structured prompt template (`prompt.py`) has all 5 skeleton components: role, context, task, format, length — PLUS negative constraint AND few-shot example (as actual text)
- [ ] `classify_intent` correctly routes a query containing "delivery" / "return" / "refund" / "membership" / "tracking" / "cancel" / "gift card" / "support hours" → `policy_question` (no LLM call in mock mode)
- [ ] `classify_intent` correctly routes an unrelated query → `general_question` (no LLM call)
- [ ] LangGraph has exactly 3 named nodes: `classify_intent`, `retrieve_and_answer`, `direct_answer`
- [ ] Conditional edge from `classify_intent` routes correctly to `retrieve_and_answer` or `direct_answer`
- [ ] Retrieval for a policy query returns chunks from the correct source document
- [ ] `retrieve_and_answer` mock output follows `"Based on the retrieved context: ..."` template
- [ ] `direct_answer` mock output is the fixed canned string
- [ ] Neither mock node makes a network call (no LLM API called when MOCK_LLM is at default)
- [ ] `AskResponse` Pydantic model has exactly: `answer` (str), `sources` (List[str]), `confidence` (float 0–1)
- [ ] Mock mode deterministically populates all 3 fields (sources = chunk IDs for policy, empty for general; confidence = 1.0)
- [ ] Retry-on-failure logic for real-LLM path is present in code (up to 2 retries)
- [ ] FastAPI app runs via `uvicorn main:app --host 0.0.0.0 --port 8000`
- [ ] Both example call JSON responses (one policy, one general) recorded in README (run with MOCK_LLM at default)
- [ ] `Dockerfile` builds successfully and serves `/ask` at port 7860 on `docker run`
- [ ] `docker build` + `docker run` commands documented in README
- [ ] README includes written RAG pipeline architecture: ingestion → embedding → retrieval → generation, naming which file/node handles each stage
- [ ] README states which stages branch on `MOCK_LLM` and what changes between mock and real-LLM state
