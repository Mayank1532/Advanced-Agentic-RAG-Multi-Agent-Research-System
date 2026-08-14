from collections import defaultdict

from agentic_rag.models import DocumentChunk, RetrievalResult


class HybridRetriever:
    """Combine vector and lexical retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_embedding,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        vector_results = self.vector_retriever.search(
            query_embedding,
            top_k=top_k,
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=top_k,
        )

        scores = defaultdict(float)
        chunks: dict[str, DocumentChunk] = {}

        for result in vector_results:
            scores[result.chunk.id] += 1.0 / (
                self.rrf_k + result.rank
            )
            chunks[result.chunk.id] = result.chunk

        for result in bm25_results:
            scores[result.chunk.id] += 1.0 / (
                self.rrf_k + result.rank
            )
            chunks[result.chunk.id] = result.chunk

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                chunk=chunks[chunk_id],
                score=float(score),
                retriever="hybrid",
                rank=rank,
                metadata=chunks[chunk_id].metadata,
            )
            for rank, (chunk_id, score) in enumerate(
                ranked,
                start=1,
            )
        ]
