import re

from rank_bm25 import BM25Okapi

from agentic_rag.models import DocumentChunk, RetrievalResult


def tokenize(text: str) -> list[str]:
    """Simple deterministic tokenizer for lexical retrieval."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Keyword retrieval using BM25."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required.")

        self.chunks = chunks
        self.corpus = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        scores = self.bm25.get_scores(tokenize(query))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[: min(top_k, len(self.chunks))]

        results: list[RetrievalResult] = []

        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(scores[index]),
                    retriever="bm25",
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        return results
