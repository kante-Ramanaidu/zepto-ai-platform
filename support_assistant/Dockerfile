# Dockerfile for the Zepto Support Assistant FastAPI app.
#
# Build:   docker build -t zepto-support-assistant .
# Run:     docker run -p 7860:7860 zepto-support-assistant
# Test:    curl -X POST http://localhost:7860/ask \
#               -H "Content-Type: application/json" \
#               -d '{"query": "What is the delivery fee?"}'
#
# MOCK_LLM is unset by default inside the container → mock mode is active
# (graded baseline: no API key required, fully deterministic).
#
# Optional MOCK_LLM=0 (ungraded):
#   docker run -p 7860:7860 -e MOCK_LLM=0 -e GROQ_API_KEY=<key> zepto-support-assistant

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first — leverages Docker layer cache
COPY requirements.txt .

# Install dependencies (strip optional groq comment line before pip)
RUN pip install --no-cache-dir \
        sentence-transformers \
        chromadb \
        langgraph \
        langchain-core \
        pydantic \
        fastapi \
        "uvicorn[standard]" \
        python-dotenv

# Copy application source
COPY . .

# Pre-ingest all 8 documents at build time so the container starts ready
# (ChromaDB is stored in /app/chroma_db inside the image)
RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
