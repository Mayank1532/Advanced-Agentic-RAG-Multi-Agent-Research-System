from pathlib import Path

from agentic_rag.rag import (
    BM25Retriever,
    DocumentLoader,
    EmbeddingService,
    HybridRetriever,
    MetadataFilter,
    TextChunker,
    VectorRetriever,
)


CORPUS_PATH = Path("data/documents/corpus.json")

MINILM_ROOT = Path(
    r"D:\HuggingFaceCache\hub"
    r"\\models--sentence-transformers--all-MiniLM-L6-v2"
    r"\snapshots"
)


def get_minilm_snapshot() -> Path:
    snapshots = list(MINILM_ROOT.iterdir())

    if not snapshots:
        raise RuntimeError("No MiniLM snapshot found.")

    return snapshots[0]


def build_retrieval_pipeline():
    documents = DocumentLoader(CORPUS_PATH).load()

    chunks = TextChunker(
        chunk_size=80,
        overlap=15,
    ).chunk_documents(documents)

    embedding_service = EmbeddingService(
        get_minilm_snapshot()
    )

    embeddings = embedding_service.encode(
        [chunk.text for chunk in chunks]
    )

    vector_retriever = VectorRetriever(
        chunks,
        embeddings,
    )

    bm25_retriever = BM25Retriever(chunks)

    hybrid_retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
    )

    return embedding_service, hybrid_retriever


def test_real_hybrid_retrieval() -> None:
    embedding_service, hybrid = build_retrieval_pipeline()

    query = "How can semantic and keyword search be combined?"

    query_embedding = embedding_service.encode([query])[0]

    results = hybrid.search(
        query=query,
        query_embedding=query_embedding,
        top_k=5,
    )

    assert results
    assert len(results) <= 5
    assert all(result.retriever == "hybrid" for result in results)

    titles = [result.chunk.title for result in results]

    assert "Hybrid Retrieval" in titles


def test_real_metadata_filtering() -> None:
    embedding_service, hybrid = build_retrieval_pipeline()

    query = "How does reranking improve candidate relevance?"

    query_embedding = embedding_service.encode([query])[0]

    results = hybrid.search(
        query=query,
        query_embedding=query_embedding,
        top_k=5,
    )

    filtered = MetadataFilter().filter(
        results,
        {"category": "reranking"},
    )

    assert filtered
    assert all(
        result.chunk.category == "reranking"
        for result in filtered
    )

    assert filtered[0].chunk.title == "Cross Encoder Reranking"
