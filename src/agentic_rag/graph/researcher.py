from pathlib import Path

from agentic_rag.graph.state import ResearchState
from agentic_rag.rag import (
    BM25Retriever,
    CrossEncoderReranker,
    DocumentLoader,
    EmbeddingService,
    HybridRetriever,
    TextChunker,
    VectorRetriever,
)


CORPUS_PATH = Path("data/documents/corpus.json")

HF_ROOT = Path(r"D:\HuggingFaceCache\hub")

EMBEDDING_ROOT = (
    HF_ROOT
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
)

RERANKER_ROOT = (
    HF_ROOT
    / "models--cross-encoder--ms-marco-MiniLM-L6-v2"
    / "snapshots"
)


class ResearchRetriever:
    """Build and reuse the complete local RAG retrieval pipeline."""

    def __init__(self) -> None:
        documents = DocumentLoader(CORPUS_PATH).load()
        chunks = TextChunker(80, 15).chunk_documents(documents)

        embedding_path = next(EMBEDDING_ROOT.iterdir())
        reranker_path = next(RERANKER_ROOT.iterdir())

        self.embedding_service = EmbeddingService(embedding_path)

        vectors = self.embedding_service.encode(
            [chunk.text for chunk in chunks]
        )

        vector_retriever = VectorRetriever(chunks, vectors)
        bm25_retriever = BM25Retriever(chunks)

        self.hybrid_retriever = HybridRetriever(
            vector_retriever,
            bm25_retriever,
        )

        self.reranker = CrossEncoderReranker(reranker_path)

    def retrieve(self, query: str) -> list[str]:
        query_vector = self.embedding_service.encode([query])[0]

        candidates = self.hybrid_retriever.search(
            query,
            query_vector,
            top_k=5,
        )

        ranked = self.reranker.rerank(
            query,
            candidates,
            top_k=3,
        )

        return [
            result.chunk.text
            for result in ranked
        ]


_retriever: ResearchRetriever | None = None


def researcher_node(state: ResearchState) -> ResearchState:
    """Retrieve evidence using hybrid search and reranking."""

    global _retriever

    if _retriever is None:
        _retriever = ResearchRetriever()

    query = state["retrieval_query"]

    evidence = _retriever.retrieve(query)

    return {
        "evidence": evidence,
    }
