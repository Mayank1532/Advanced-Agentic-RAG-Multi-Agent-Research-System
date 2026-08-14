from typing import Protocol

from agentic_rag.models import RetrievalResult


class Reranker(Protocol):
    """Interface for candidate reranking."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        ...
