import numpy as np

from agentic_rag.models import DocumentChunk, RetrievalResult
from agentic_rag.rag.bm25 import BM25Retriever
from agentic_rag.rag.hybrid import HybridRetriever
from agentic_rag.rag.metadata import MetadataFilter
from agentic_rag.rag.vector import VectorRetriever


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            id="a",
            text="BM25 keyword retrieval matches exact technical terms.",
            source="retrieval",
            title="BM25",
            category="retrieval",
            chunk_index=0,
        ),
        DocumentChunk(
            id="b",
            text="Vector search retrieves semantically similar information.",
            source="retrieval",
            title="Vector",
            category="retrieval",
            chunk_index=0,
        ),
        DocumentChunk(
            id="c",
            text="CrossEncoder reranking improves candidate relevance.",
            source="reranking",
            title="Reranking",
            category="reranking",
            chunk_index=0,
        ),
    ]


def test_bm25_returns_keyword_match_first() -> None:
    retriever = BM25Retriever(make_chunks())

    results = retriever.search("BM25 keyword retrieval", top_k=2)

    assert results[0].chunk.id == "a"
    assert results[0].retriever == "bm25"


def test_vector_retriever_returns_expected_candidate() -> None:
    chunks = make_chunks()

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )

    retriever = VectorRetriever(chunks, embeddings)

    results = retriever.search(
        np.asarray([0.0, 1.0], dtype=np.float32),
        top_k=2,
    )

    assert results[0].chunk.id == "b"
    assert results[0].retriever == "vector"


def test_hybrid_fusion_combines_retrievers() -> None:
    chunks = make_chunks()

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )

    vector = VectorRetriever(chunks, embeddings)
    bm25 = BM25Retriever(chunks)

    hybrid = HybridRetriever(vector, bm25)

    results = hybrid.search(
        "BM25 keyword retrieval",
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=3,
    )

    assert results
    assert results[0].retriever == "hybrid"
    assert {result.chunk.id for result in results} == {"a", "b", "c"}


def test_metadata_filter_keeps_only_requested_category() -> None:
    results = [
        RetrievalResult(
            chunk=chunk,
            score=1.0,
            retriever="hybrid",
            rank=index,
        )
        for index, chunk in enumerate(make_chunks(), start=1)
    ]

    filtered = MetadataFilter().filter(
        results,
        {"category": "reranking"},
    )

    assert len(filtered) == 1
    assert filtered[0].chunk.id == "c"
    assert filtered[0].rank == 1


def test_metadata_filter_without_filters_is_noop() -> None:
    results = [
        RetrievalResult(
            chunk=make_chunks()[0],
            score=1.0,
            retriever="bm25",
            rank=1,
        )
    ]

    assert MetadataFilter().filter(results) == results
