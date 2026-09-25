# Zepto Data & AI Platform

## Project Overview

This capstone builds one connected AI/ML platform across three modules. A data-engineering pipeline (`/data_pipeline`) scrapes raw product data from a public catalogue, cleans it, converts currencies, and loads it into a normalized SQLite relational store. An analytics pipeline (`/analytics`) profiles the classic Titanic dataset end to end — cleaning, visualizing, and building a full predictive-modeling suite with rigorous evaluation. A GenAI support assistant (`/support_assistant`) embeds Zepto's policy documents, routes queries through a LangGraph graph, and serves grounded answers via a FastAPI endpoint — all fully operational with no API key required (mock mode is the graded default).

---

## Repository Structure

```
zepto-ai-platform/
├── README.md                        ← This file
├── data_pipeline/
│   ├── data_pipeline.ipynb          ← Scrape → clean → convert → SQLite → queries
│   ├── books.db                     ← SQLite database (committed)
│   ├── requirements.txt
│   └── README.md
├── analytics/
│   ├── 01_eda.ipynb                 ← Load, profile, clean, EDA, save titanic.csv
│   ├── 02_modeling.ipynb            ← ML pipeline, tuning, regression, joblib save
│   ├── titanic.csv                  ← Offline fallback (committed)
│   ├── best_pipeline.joblib         ← Saved full sklearn Pipeline
│   ├── requirements.txt
│   └── README.md
└── support_assistant/
    ├── docs/                        ← 8 Zepto policy documents
    │   ├── doc_01.txt … doc_08.txt
    ├── ingest.py                    ← Embed docs into ChromaDB
    ├── graph.py                     ← LangGraph StateGraph (3 nodes)
    ├── prompt.py                    ← Structured prompt template
    ├── schemas.py                   ← Pydantic request/response models
    ├── main.py                      ← FastAPI app
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

---

## Setup & Installation

Each module has its own `requirements.txt`. Install per module before running it.

### Module 1 — Data Pipeline
```bash
cd data_pipeline
pip install -r requirements.txt
```

### Module 2 — Analytics
```bash
cd analytics
pip install -r requirements.txt
```

### Module 3 — Support Assistant
```bash
cd support_assistant
pip install -r requirements.txt
```

---

## How to Run Each Module

### Module 1 — Data Pipeline
```bash
cd data_pipeline
jupyter notebook data_pipeline.ipynb
# Run all cells top to bottom — scrapes books.toscrape.com, builds books.db, runs all queries
```

### Module 2 — Analytics
```bash
cd analytics
# Step 1: EDA notebook (generates titanic.csv — must run first)
jupyter notebook 01_eda.ipynb

# Step 2: Modeling notebook (reads titanic.csv — run after Step 1)
jupyter notebook 02_modeling.ipynb
```

### Module 3 — Support Assistant
```bash
cd support_assistant

# Step 1: Ingest documents into ChromaDB (run once)
python ingest.py

# Step 2: Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000

# Step 3: Test the endpoint (MOCK_LLM at default — no API key needed)
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the delivery fee for orders below INR 149?\"}"
```

#### Docker (alternative)
```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# Endpoint available at http://localhost:7860/ask
```

---

## Currency Conversion Rate (Module 1)

**Fixed baseline: 1 GBP = 105.50 INR**

This is an artificial, project-defined constant. It requires no API call and no date reference. All `price_inr` values in `books.db` are computed using exactly this rate.

---

## Design Decisions

### Module 1 — Data Pipeline
- **Scope**: Scraped books across 3+ categories from books.toscrape.com (≥ 60 books total).
- **Price NaN → median imputation**: Price is a numeric field and median is robust to outliers. Only a very small number of rows would be affected.
- **Rating NaN → drop row**: The star rating uses an ordinal text scale (One–Five). Median imputation on an ordinal field is unreliable, and the number of unparseable rows is negligible.
- **Two-table schema**: `categories` and `books` are separated so that category names are stored once (normalized), with `books.category_id` as a foreign key. This avoids string duplication and enables clean JOINs.

### Module 2 — Analytics
- **Missing value decisions**: deck (77.22% → drop column), age (19.87% → median imputation), embarked/embark_town (0.22% → drop rows). All decisions cite the <5%/5–30%/>30% threshold rule.
- **Stratified split**: Class imbalance (~38% survived) means a random split could skew the class distribution. Stratification ensures both train and test sets reflect the true ratio.
- **ColumnTransformer + Pipeline**: Enforces fit-on-train-only preprocessing structurally — the Pipeline's `.fit()` on training data and `.transform()` on test data is the architectural guarantee against leakage.
- **Imbalance strategy**: SMOTE achieves the best F1 (0.7774) by oversampling the minority class in training only via `imblearn.pipeline.Pipeline`. `class_weight='balanced'` is simpler and achieves competitive Recall (0.8116).
- **Model recommendation**: Random Forest (F1=0.7442, AUC=0.8287) is recommended. For deployment where missing a survivor is costly, the `class_weight='balanced'` variant (Recall=0.8116) is preferred.

### Module 3 — Support Assistant
- **Per-document chunking**: The 8 policy documents are short enough that one chunk per document is appropriate. Smaller fixed-size chunks would fragment sentences without improving retrieval quality for this corpus size.
- **MOCK_LLM=1 (default, graded)**: Fully deterministic. No API key, no network calls. `classify_intent` uses a keyword heuristic; `retrieve_and_answer` returns a canned template from the top retrieved chunk; `direct_answer` returns a fixed string.
- **MOCK_LLM=0 (optional, ungraded)**: Routes generation to Groq's free-tier LLM. This path is present in code but does not affect the graded submission.
- **Dockerfile pre-ingests at build time**: Running `python ingest.py` inside the `RUN` layer means the container starts with ChromaDB already populated — no startup delay.

---

## Example API Calls — Module 3 (MOCK_LLM at default)

**Call 1 — Policy question (triggers retrieval):**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the delivery fee for orders below INR 149?\"}"
```
**Raw JSON response (recorded from live server):**
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}
```

**Call 2 — General question (no retrieval):**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the capital of France?\"}"
```
**Raw JSON response (recorded from live server):**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## Git Workflow

This repository uses feature branches for development work, with merge commits back into `main`. Three feature branches were created and merged:
- `feature/data-pipeline` — 2 commits, merged into `main`
- `feature/analytics` — 1 commit, merged into `main`
- `feature/support-assistant` — 2 commits, merged into `main`

The branch history is visible via:

```bash
git log --graph --all --oneline
```
