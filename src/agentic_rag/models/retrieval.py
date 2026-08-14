from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.models import DocumentChunk


class RetrievalResult(BaseModel):
    """A document chunk returned by a retrieval stage."""

    chunk: DocumentChunk
    score: float
    retriever: str
    rank: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
