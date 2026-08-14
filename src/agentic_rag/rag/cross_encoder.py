from pathlib import Path

from sentence_transformers import CrossEncoder

from agentic_rag.models import RetrievalResult


class CrossEncoderReranker:
    """Rerank retrieved candidates using a local CrossEncoder model."""

    def __init__(self, model_path: Path) -> None:
        self.model = CrossEncoder(
            str(model_path),
            local_files_only=True,
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not results:
            return []

        pairs = [
            [query, result.chunk.text]
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]

        return [
            result.model_copy(
                update={
                    "score": float(score),
                    "retriever": "reranker",
                    "rank": rank,
                }
            )
            for rank, (result, score) in enumerate(
                ranked,
                start=1,
            )
        ]
