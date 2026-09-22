# schemas.py
# Pydantic request and response models for the FastAPI /ask endpoint.
# These are used in both mock mode (MOCK_LLM=1, graded default) and
# the optional real-LLM mode (MOCK_LLM=0).

from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    """Request body for POST /ask."""
    query: str = Field(..., description="The user's question")


class AskResponse(BaseModel):
    """
    Validated response returned by every /ask call.

    Fields:
        answer     — The answer text (canned template in mock mode).
        sources    — List of chunk/document IDs used during retrieval.
                     Empty list for general_question answers.
        confidence — Confidence score in [0.0, 1.0].
                     Fixed at 1.0 in mock mode (deterministic).
    """
    answer:     str       = Field(..., description="Answer to the user's query")
    sources:    List[str] = Field(
        default_factory=list,
        description="Document IDs used (empty for general questions)"
    )
    confidence: float     = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
