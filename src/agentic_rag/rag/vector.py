import faiss
import numpy as np

from agentic_rag.models import DocumentChunk, RetrievalResult


class VectorRetriever:
    """Semantic retrieval using normalized embeddings and FAISS."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")

        self.chunks = chunks

        if len(embeddings) == 0:
            raise ValueError("At least one embedding is required.")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(
            np.asarray(embeddings, dtype=np.float32)
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query,
            min(top_k, len(self.chunks)),
        )

        results: list[RetrievalResult] = []

        for rank, (score, index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    retriever="vector",
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        return results
