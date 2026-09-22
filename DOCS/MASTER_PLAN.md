# MASTER PLAN — Zepto Data & AI Platform Capstone
## Total: 100 Marks | 3 Modules | 1 Repository

---

## Repository Structure (Final)

```
zepto-ai-platform/                      ← Single public GitHub repo root
│
├── README.md                           ← ROOT README (required by spec)
│
├── data_pipeline/                      ← Module 1 — 25 marks
│   ├── data_pipeline.ipynb             ← Scrape + clean + DB + queries
│   ├── books.db                        ← SQLite database (committed)
│   ├── requirements.txt
│   └── README.md
│
├── analytics/                          ← Module 2 — 50 marks
│   ├── 01_eda.ipynb                    ← Part A: load, profile, clean, EDA
│   ├── 02_modeling.ipynb               ← Part B: ML pipeline, tuning, regression
│   ├── titanic.csv                     ← Offline fallback (MUST be committed)
│   ├── best_pipeline.joblib            ← Saved full sklearn Pipeline
│   ├── requirements.txt
│   └── README.md
│
└── support_assistant/                  ← Module 3 — 25 marks
    ├── docs/
    │   ├── doc_01.txt … doc_08.txt     ← 8 policy documents
    ├── ingest.py
    ├── graph.py
    ├── prompt.py
    ├── schemas.py
    ├── main.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

---

## Phase-by-Phase Build Order

Work in this order to avoid context-switching and to nail the git workflow requirement in one go.

```
Phase 0  →  Git repo setup + branch strategy
Phase 1  →  Module 1: Data Pipeline          (25 marks)
Phase 2  →  Module 2 Part A: EDA             (analytics notebook 1)
Phase 3  →  Module 2 Part B: Modeling        (analytics notebook 2)
Phase 4  →  Module 3: Support Assistant      (25 marks)
Phase 5  →  Root README + final checks
Phase 6  →  Git merge + submission
```

---

## Phase 0 — Git Repository Setup

### 0.1 — Create the repo
```bash
# On GitHub: create a NEW PUBLIC repository named "zepto-ai-platform"
# Clone it locally:
git clone https://github.com/YOUR_USERNAME/zepto-ai-platform.git
cd zepto-ai-platform
```

### 0.2 — Create the folder skeleton
```bash
mkdir data_pipeline analytics support_assistant
mkdir support_assistant\docs
# Create placeholder READMEs so git tracks the folders
echo "# Data Pipeline" > data_pipeline\README.md
echo "# Analytics"     > analytics\README.md
echo "# Support Assistant" > support_assistant\README.md
echo "# Zepto AI Platform" > README.md
```

### 0.3 — First commit on main
```bash
git add .
git commit -m "chore: initialize repo structure with module folders"
git push -u origin main
```

### 0.4 — Create the feature branch (REQUIRED FOR MARKS)
> The spec requires: one feature branch, committed to ≥ 2 times, merged back into main.
> This is checked once for the WHOLE repo. Do ALL your work on this branch.

```bash
git checkout -b feature/zepto-platform
# All development work below happens on this branch
```

---

## Phase 1 — Module 1: Data Pipeline

**Reference file**: `DOCS/MODULE_1_DATA_PIPELINE.md`

### Steps (in order):

| Step | Task | File |
|------|------|------|
| 1.1 | Create `data_pipeline/requirements.txt` | `data_pipeline/requirements.txt` |
| 1.2 | Scrape ≥ 60 books from ≥ 3 categories (requests + BeautifulSoup) | `data_pipeline.ipynb` Cell 1 |
| 1.3 | Clean fields → `price_gbp` (float), `rating` (int), `in_stock` (bool) | Cell 2 |
| 1.4 | Add `price_inr = price_gbp × 105.50` | Cell 3 |
| 1.5 | Create SQLite schema: `categories` + `books` (PK/FK) | Cell 4 |
| 1.6 | Insert all rows into SQLite | Cell 5 |
| 1.7 | Write + execute ≥ 5 SQL queries (WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN/IN, JOIN) | Cell 6 |
| 1.8 | `pd.read_sql` vs `pd.merge` side-by-side comparison | Cell 7 |
| 1.9 | Write `data_pipeline/README.md` (install, run, fixed rate, design decisions) | README |
| 1.10 | Commit progress to feature branch | git |

### Commit after Phase 1:
```bash
git add data_pipeline/
git commit -m "feat: complete data pipeline module — scrape, clean, SQLite, queries"
```

---

## Phase 2 — Module 2 Part A: EDA (`01_eda.ipynb`)

**Reference file**: `DOCS/MODULE_2_ANALYTICS.md` — Steps 1–6

| Step | Task | Acceptance criterion |
|------|------|---------------------|
| 2.1 | `sns.load_dataset('titanic')` → profile → `df.to_csv("titanic.csv")` | `titanic.csv` committed; loaded ONCE |
| 2.2 | Report missing % per column; apply threshold rules (drop rows <5%, impute 5–30%, drop column >30%) | Exact % + rule stated in README |
| 2.3 | Histogram + box plot for `age` and `fare`; IQR outlier counts for both | Counts reported in README |
| 2.4 | Mean/median/mode for `fare`; state skewness direction with evidence | Mean > median > mode → right-skewed |
| 2.5 | Survival rates by sex, pclass, sex+pclass (boolean masking) | Numeric rates in README |
| 2.6 | Correlation matrix on EXACTLY 6 cols (`survived, pclass, age, sibsp, parch, fare`) — heatmap | adult_male and alone excluded |
| 2.7 | Name top 2 off-diagonal pairs by abs(corr); interpret both in text | Written interpretation |
| 2.8 | ≥ 4 multivariate charts; 2–4 sentence written interpretation per chart | All interpretations in README/notebook |
| 2.9 | EDA standardization check: z-score on `age` and `fare`; before/after comparison | Shows mean≈0, std≈1 |

### Commit after Phase 2:
```bash
git add analytics/01_eda.ipynb analytics/titanic.csv
git commit -m "feat: EDA notebook — profiling, cleaning, visualizations, standardization check"
```

---

## Phase 3 — Module 2 Part B: Modeling (`02_modeling.ipynb`)

**Reference file**: `DOCS/MODULE_2_ANALYTICS.md` — Steps 7–16

| Step | Task | Acceptance criterion |
|------|------|---------------------|
| 3.1 | `pd.read_csv("titanic.csv")` — NEVER call `sns.load_dataset` again | Dataset loaded from CSV only |
| 3.2 | Stratified train/test split (justify class balance) | Written justification |
| 3.3 | ColumnTransformer + Pipeline: median impute → StandardScaler (num); most_freq impute → OHE (cat) | Fit on train ONLY |
| 3.4 | Train Logistic Regression, Decision Tree, Random Forest | Same split for all 3 |
| 3.5 | `plot_tree` visualization of Decision Tree (labeled features + class names) | Saved as `.png` |
| 3.6 | Full metrics per model: confusion matrix, accuracy, precision, recall, F1, ROC/AUC | Side-by-side table |
| 3.7 | Imbalance comparison: baseline vs `class_weight='balanced'` vs SMOTE (train fold only) | Written conclusion |
| 3.8 | `GridSearchCV` over RF `n_estimators, max_depth, max_features` + report OOB score | `oob_score=True` at construction |
| 3.9 | Linear regression for `fare`: MAE, RMSE, R², Adjusted R² + residual plot + heteroscedasticity conclusion | All 4 metrics + written conclusion |
| 3.10 | Model comparison table: classifiers table + regression table as SEPARATE groups | Not merged into one scale |
| 3.11 | Written recommendation ≥ 3 sentences citing specific metric values | In README |
| 3.12 | `joblib.dump(full_pipeline, "best_pipeline.joblib")` — preprocessing + estimator together | Reload verified on raw input |

### Commit after Phase 3:
```bash
git add analytics/02_modeling.ipynb analytics/best_pipeline.joblib
git commit -m "feat: modeling notebook — 3 classifiers, SMOTE, GridSearchCV, regression, joblib pipeline"
```

---

## Phase 4 — Module 3: Support Assistant

**Reference file**: `DOCS/MODULE_3_SUPPORT_ASSISTANT.md`

| Step | Task | File |
|------|------|------|
| 4.1 | Create `support_assistant/requirements.txt` | requirements.txt |
| 4.2 | Write all 8 `docs/doc_0X.txt` files with exact spec text | docs/ |
| 4.3 | Write `schemas.py` — `AskRequest` + `AskResponse` Pydantic models | schemas.py |
| 4.4 | Write `ingest.py` — load docs, embed with `all-MiniLM-L6-v2`, store in ChromaDB | ingest.py |
| 4.5 | Run `python ingest.py` — verify 8 docs ingested | chroma_db/ |
| 4.6 | Write `prompt.py` — 5-skeleton prompt with negative constraint + few-shot example | prompt.py |
| 4.7 | Write `graph.py` — LangGraph StateGraph: 3 nodes + conditional edge + MOCK_LLM toggle | graph.py |
| 4.8 | Write `main.py` — FastAPI `POST /ask` endpoint | main.py |
| 4.9 | Run `uvicorn main:app --port 8000` and test both example calls | terminal |
| 4.10 | Record raw JSON responses in README (MOCK_LLM at default) | README |
| 4.11 | Write `Dockerfile` | Dockerfile |
| 4.12 | `docker build` + `docker run` locally — verify `/ask` works | terminal |
| 4.13 | Write `support_assistant/README.md` (RAG architecture description + transcripts + docker cmds) | README |

### Commit after Phase 4:
```bash
git add support_assistant/
git commit -m "feat: support assistant — ChromaDB ingestion, LangGraph RAG, FastAPI, Dockerfile"
```

---

## Phase 5 — Root README

The root `README.md` must cover all three modules. Structure it like this:

```markdown
# Zepto Data & AI Platform

## Project Overview
[One paragraph connecting all 3 modules as one story]

## Repository Structure
[Folder tree]

## Setup & Installation
### Module 1 — Data Pipeline
pip install -r data_pipeline/requirements.txt

### Module 2 — Analytics
pip install -r analytics/requirements.txt

### Module 3 — Support Assistant
pip install -r support_assistant/requirements.txt

## How to Run Each Module

### Module 1
cd data_pipeline
jupyter notebook data_pipeline.ipynb

### Module 2
cd analytics
jupyter notebook 01_eda.ipynb   # run first — generates titanic.csv
jupyter notebook 02_modeling.ipynb

### Module 3
cd support_assistant
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 8000

## Currency Conversion Rate (Module 1)
Fixed baseline: 1 GBP = 105.50 INR
(Project-defined constant — no API, no date reference)

## Design Decisions

### Module 1
- Price NaN → median imputation (outlier-robust)
- Rating NaN → drop row (rare; ordinal imputation unreliable)
- Two-table SQLite schema (categories ↔ books) for normalized PK/FK structure

### Module 2
- [Missing value decisions with exact % and threshold rule for each column]
- Stratified split because class imbalance (~38% survived) would distort metrics
- ColumnTransformer+Pipeline enforces fit-on-train-only preprocessing structurally
- [Imbalance strategy conclusion]
- [Model recommendation with metric values]

### Module 3
- Per-document chunking (8 short policy docs — no need for smaller chunks)
- MOCK_LLM=1 (default): fully deterministic, no API key, no network calls — graded path
- MOCK_LLM=0 (optional): routes to Groq free tier LLM — ungraded extension
- Dockerfile pre-ingests docs at build time for instant startup
```

### Commit after Phase 5:
```bash
git add README.md
git commit -m "docs: complete root README with setup, run instructions, design decisions"
```

---

## Phase 6 — Git Merge + Final Submission

### 6.1 — Verify the feature branch has ≥ 2 commits
```bash
git log --oneline feature/zepto-platform
# Should show at least 2 commits on this branch (you'll have 4+)
```

### 6.2 — Push the feature branch
```bash
git push -u origin feature/zepto-platform
```

### 6.3 — Merge into main (creates a merge commit — visible in git log --graph)
```bash
git checkout main
git merge --no-ff feature/zepto-platform -m "merge: zepto-platform feature branch into main"
git push origin main
```

### 6.4 — Verify with git log
```bash
git log --graph --all --oneline
# Should show: branch created from main, ≥ 2 commits on feature branch, merge commit back to main
```

### 6.5 — Final repo check before submitting
```bash
git log --oneline          # see all commits
git branch -a              # see feature branch exists in history
```

---

## Root-Level Requirements Cross-Check

| Requirement | Where it lives | Status |
|---|---|---|
| Single public GitHub repo | Root | ✅ Create once |
| Root `README.md` | Root | Phase 5 |
| Install steps per module | Root README + module READMEs | Phase 5 |
| Run steps per module | Root README | Phase 5 |
| Design decisions | Root README + module READMEs | Each phase |
| No screenshots/PDFs/slides | N/A | ✅ All text/code/md |
| Feature branch + ≥ 2 commits + merged | git history | Phase 6 |
| No paid services | N/A | ✅ All free/local |

---

## Full Marks (100/100) Final Checklist

### Module 1 — Data Pipeline (25/25)
- [ ] ≥ 60 books scraped, ≥ 3 categories, runs end-to-end
- [ ] `price_gbp` (float), `rating` (1–5 int), `in_stock` (bool), `price_inr` (float) — correct types
- [ ] `price_inr = price_gbp × 105.50` — exact rate in README
- [ ] `books.db` committed OR recreation script present
- [ ] Two-table PK/FK schema (`categories` ↔ `books`)
- [ ] ≥ 5 SQL queries with output: WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN/IN, JOIN
- [ ] `pd.read_sql` + `pd.merge` both shown, confirmed equivalent
- [ ] `data_pipeline/README.md` covers install, run, rate, cleaning decisions

### Module 2 — Analytics (50/50)
- [ ] `titanic.csv` committed in `/analytics`
- [ ] `sns.load_dataset('titanic')` called exactly once (never again in modeling)
- [ ] Missing % per column + threshold-rule justification for each
- [ ] IQR outlier counts for `age` and `fare`
- [ ] Fare skewness: mean/median/mode values + right-skewed conclusion
- [ ] Survival rates by sex, pclass, sex+pclass (numeric values)
- [ ] Correlation matrix: exactly 6 cols, `adult_male`/`alone` excluded
- [ ] Top 2 off-diagonal correlations named + interpreted
- [ ] ≥ 4 multivariate charts, each with 2–4 sentence written interpretation
- [ ] EDA standardization before/after for `age` and `fare`
- [ ] Stratified split with written justification
- [ ] All preprocessing fit on train ONLY (Pipeline/ColumnTransformer)
- [ ] 3 classifiers on identical split
- [ ] `plot_tree` with labeled feature names + class names
- [ ] Confusion matrix, accuracy, precision, recall, F1, ROC/AUC per classifier
- [ ] 3-way imbalance comparison + written conclusion; SMOTE on train fold only
- [ ] `GridSearchCV` best params + OOB score (`oob_score=True` at construction)
- [ ] Regression: MAE, RMSE, R², Adjusted R² + residual plot + heteroscedasticity conclusion
- [ ] Model comparison table: classifier metrics + regression metrics as SEPARATE groups
- [ ] Final recommendation ≥ 3 sentences citing specific metric values
- [ ] `best_pipeline.joblib` = preprocessing + estimator; reloaded, verified on raw input

### Module 3 — Support Assistant (25/25)
- [ ] All 8 `docs/doc_0X.txt` files with exact spec text
- [ ] `python ingest.py` loads all 8 docs into ChromaDB
- [ ] Prompt template: all 5 skeleton components + negative constraint + few-shot example (actual text)
- [ ] `classify_intent` keyword heuristic routes policy keywords → `policy_question` (no LLM call)
- [ ] `classify_intent` routes unrelated query → `general_question` (no LLM call)
- [ ] LangGraph: 3 nodes + conditional edge working correctly
- [ ] Retrieval returns chunks from correct source document for policy query
- [ ] `retrieve_and_answer` mock: `"Based on the retrieved context: ..."` template
- [ ] `direct_answer` mock: fixed canned string
- [ ] No network calls in mock mode (MOCK_LLM at default)
- [ ] `AskResponse(answer, sources, confidence)` deterministically populated in mock mode
- [ ] Retry logic (up to 2 retries) for real-LLM path in code
- [ ] FastAPI app runs; both example call JSON responses recorded in README
- [ ] `Dockerfile` builds + runs locally, serves `/ask` on port 7860
- [ ] `docker build` + `docker run` documented in README
- [ ] RAG architecture description: ingestion → embedding → retrieval → generation (which file/node handles each)
- [ ] README states which stages branch on `MOCK_LLM` and what changes

### Repository & Git (Required for Module 1 marks)
- [ ] Feature branch created from main
- [ ] Feature branch committed to ≥ 2 times
- [ ] Feature branch merged back into main (visible in `git log --graph --all`)
- [ ] Single public GitHub repo submitted (not 3 separate repos)

---

## Dependencies Quick Reference

### Module 1
```
requests
beautifulsoup4
pandas
```

### Module 2
```
seaborn
pandas
matplotlib
scikit-learn
imbalanced-learn
joblib
```

### Module 3
```
sentence-transformers
chromadb
langgraph
langchain-core
pydantic
fastapi
uvicorn[standard]
python-dotenv
```

---

## Important Rules — Never Break These

| Rule | Why it matters |
|---|---|
| `1 GBP = 105.50 INR` — hard-coded constant, no API | Grader checks this exact value |
| `sns.load_dataset('titanic')` called exactly ONCE | Spec requirement; grader checks no second call |
| `titanic.csv` committed in `/analytics` | Grader uses `pd.read_csv("titanic.csv")` to assess your work offline |
| `adult_male` and `alone` excluded from correlation matrix | Spec explicitly calls these out as derived flags |
| All preprocessing fit on train ONLY | Any fit on test data = data leakage = marks deducted |
| `oob_score=True` passed at RandomForest construction | Without it `oob_score_` won't be populated |
| `SMOTE` applied only to training fold | Applying to full data before split = leakage |
| Save full Pipeline (preprocessor + estimator) with joblib | Bare estimator alone does not satisfy the requirement |
| `MOCK_LLM` unset or `=1` for all graded Module 3 output | Real LLM calls are optional and ungraded |
| Prompt template as actual text, not just described | Spec says "as actual text (not just described)" |
| README example calls recorded with MOCK_LLM at default | Spec explicitly states this |
| Single public repo, NOT 3 separate repos | Explicitly forbidden by spec |
| No PDFs, slides, screenshots as deliverables | All written content in Markdown/notebook cells |
