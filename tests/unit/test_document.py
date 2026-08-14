import pytest
from pydantic import ValidationError

from agentic_rag.models import DocumentChunk


def test_document_chunk_creation() -> None:
    chunk = DocumentChunk(
        id="doc-001-0",
        text="Hybrid retrieval combines multiple retrieval strategies.",
        source="rag-basics",
        title="Hybrid Retrieval",
        category="retrieval",
        chunk_index=0,
    )

    assert chunk.id == "doc-001-0"
    assert chunk.chunk_index == 0
    assert chunk.metadata == {}


def test_document_chunk_preserves_metadata() -> None:
    chunk = DocumentChunk(
        id="doc-002-0",
        text="Reranking improves candidate ordering.",
        source="rag-advanced",
        title="Reranking",
        category="retrieval",
        chunk_index=0,
        metadata={"difficulty": "advanced", "year": 2026},
    )

    assert chunk.metadata["difficulty"] == "advanced"
    assert chunk.metadata["year"] == 2026


def test_document_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            id="doc-003-0",
            text="",
            source="test",
            title="Test",
            category="test",
            chunk_index=0,
        )


def test_document_chunk_rejects_negative_chunk_index() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            id="doc-004-0",
            text="Some text",
            source="test",
            title="Test",
            category="test",
            chunk_index=-1,
        )
