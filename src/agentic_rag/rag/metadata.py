from typing import Any

from agentic_rag.models import RetrievalResult


class MetadataFilter:
    """Apply deterministic metadata constraints to retrieval results."""

    def filter(
        self,
        results: list[RetrievalResult],
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if not filters:
            return results

        filtered: list[RetrievalResult] = []

        for result in results:
            matches = all(
                result.chunk.category == value
                if key == "category"
                else result.chunk.source == value
                if key == "source"
                else result.chunk.metadata.get(key) == value
                for key, value in filters.items()
            )

            if matches:
                filtered.append(result)

        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(filtered, start=1)
        ]
