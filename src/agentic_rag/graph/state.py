from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    retrieval_query: str
    evidence: list[str]
    analysis: str
    draft: str
    review: str
    approved: bool
    revision_count: int
