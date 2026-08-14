from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """A complete source document before chunking."""

    id: str
    title: str
    source: str
    category: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
