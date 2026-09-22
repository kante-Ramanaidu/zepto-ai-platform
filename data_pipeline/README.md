# Module 1 — Data Pipeline

## Overview

Scrapes product data from [books.toscrape.com](https://books.toscrape.com) (a public scraping-practice site — no login, no API key), cleans the raw fields into proper types, converts GBP prices to INR using a fixed project-defined rate, loads everything into a normalized two-table SQLite database, and demonstrates both SQL queries and pandas operations on the result.

---

## Currency Conversion Rate

**Fixed baseline: 1 GBP = 105.50 INR**

This is a project-defined constant — not a live or historical market rate. No API call or network access is needed. Every `price_inr` value in `books.db` is computed as `price_gbp × 105.50`.

---

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `requests`, `beautifulsoup4`, `pandas` (all free, no API keys required).

---

## Run

```bash
jupyter notebook data_pipeline.ipynb
```

Run all cells top to bottom. The notebook:
1. Scrapes ≥ 60 books across ≥ 3 categories from books.toscrape.com
2. Cleans fields into proper types
3. Converts GBP → INR at the fixed rate above
4. Creates `books.db` (SQLite) with the two-table normalized schema
5. Executes ≥ 5 SQL queries with printed output
6. Shows `pd.read_sql` and `pd.merge` side-by-side equivalence

The `books.db` file is committed to the repository — you can also regenerate it from scratch by running the notebook end to end.

---

## Database Schema

Two tables with a primary/foreign key relationship:

```sql
categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT    UNIQUE NOT NULL
)

books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL,   -- 1 to 5
    in_stock    INTEGER NOT NULL,   -- 0 (False) or 1 (True)
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
)
```

---

## Cleaning & Design Decisions

### price_gbp
Strip the `£` currency symbol and cast to `float`. If the result is `NaN` (unexpected format), **impute with the median price** of the successfully parsed rows.

**Justification**: Price is a continuous numeric field. Median imputation is robust to the right-skew that price distributions typically show, and only a negligible number of rows (if any) would require it. Dropping them would be equally valid, but imputation preserves every row.

### rating
Map text labels (`One` → 1, `Two` → 2, `Three` → 3, `Four` → 4, `Five` → 5) to integers. If the class attribute contains an unexpected value, **drop that row**.

**Justification**: The rating is an ordinal integer (1–5). There is no sensible numeric median to impute on a mapping failure — the mapping itself broke, meaning the source HTML is malformed. The number of such rows is expected to be zero or near-zero, so dropping them is the safest and cleanest choice.

### in_stock
Check whether the availability text contains `"in stock"` (case-insensitive) → `True/1`, else `False/0`. No imputation needed — the field is always present in the HTML.

### Two-table schema rationale
Separating `categories` from `books` avoids repeating the category name string on every row. `category_id` as a foreign key keeps the schema in 2NF, enables efficient GROUP BY / JOIN queries, and mirrors how a production catalog database would be structured.

---

## SQL Queries Demonstrated

| Query | Clauses covered |
|-------|----------------|
| Books cheaper than £15, sorted by price | SELECT, WHERE, ORDER BY |
| Top 10 most expensive books | ORDER BY DESC, LIMIT |
| All distinct rating values present | DISTINCT |
| Books priced between £20 and £40 | BETWEEN |
| Books rated 4 or 5 stars | IN |
| Top 10 highest-rated books with category name | JOIN (books ↔ categories) |

---

## pd.read_sql vs pd.merge

The JOIN query result is reproduced two ways and shown side by side:
- `pd.read_sql(sql_join, conn)` — SQL-side join
- `pd.merge(df_books, df_categories, on="category_id")` — Python-side merge

Both outputs are confirmed equivalent via `pd.testing.assert_frame_equal`.
