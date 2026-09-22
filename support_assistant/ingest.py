# ingest.py
# Loads all 8 Zepto policy documents from docs/, embeds each one using the
# all-MiniLM-L6-v2 sentence-transformers model (local, no API key), and stores
# the embeddings in a persistent ChromaDB collection.
#
# Run once before starting the FastAPI server:
#   python ingest.py

import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_DIR       = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH    = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL    = "all-MiniLM-L6-v2"


def ingest():
    """
    Ingestion pipeline:
      1. Read each doc_0X.txt from docs/
      2. Embed with all-MiniLM-L6-v2 (local, 384-dim vectors)
      3. Store in ChromaDB persistent collection 'zepto_policies'

    Uses simple per-document chunking — one chunk per file.
    The 8 documents are short policy paragraphs; no sub-chunking is needed.
    """
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate for a clean re-ingestion (idempotent)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for retrieval
    )
    print(f"Created collection '{COLLECTION_NAME}'")

    # Collect all doc files in sorted order
    doc_files = sorted(
        f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")
    )

    if not doc_files:
        raise FileNotFoundError(f"No .txt files found in {DOCS_DIR}")

    documents, ids, embeddings, metadatas = [], [], [], []

    for doc_file in doc_files:
        doc_id   = doc_file.replace(".txt", "")          # e.g. "doc_01"
        filepath = os.path.join(DOCS_DIR, doc_file)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"  WARNING: {doc_file} is empty — skipping")
            continue

        # Embed (local inference, no network call)
        embedding = model.encode(text, normalize_embeddings=True).tolist()

        documents.append(text)
        ids.append(doc_id)
        embeddings.append(embedding)
        metadatas.append({"source": doc_file, "doc_id": doc_id})

        print(f"  Embedded: {doc_id}  ({len(text)} chars)")

    # Batch upsert into ChromaDB
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"\n✅ Ingested {len(ids)} documents into ChromaDB collection '{COLLECTION_NAME}'")
    print(f"   Collection count: {collection.count()}")


if __name__ == "__main__":
    ingest()
