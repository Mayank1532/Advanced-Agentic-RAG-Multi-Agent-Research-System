from typing import Protocol


class QueryRewriter(Protocol):
    """Interface for converting a user question into a retrieval query."""

    def rewrite(self, query: str) -> str:
        ...
