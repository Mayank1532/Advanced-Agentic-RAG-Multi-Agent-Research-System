from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A retrievable chunk of a source document."""

    id: str
    text: str = Field(min_length=1)
    source: str
    title: str
    category: str
    chunk_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
