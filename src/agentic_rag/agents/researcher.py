from agentic_rag.graph.researcher import ResearchRetriever


class ResearcherAgent:
    """Agent responsible for information gathering."""

    def __init__(self, retriever: ResearchRetriever) -> None:
        self.retriever = retriever

    def research(self, query: str) -> list[str]:
        if not query.strip():
            raise ValueError("Research query cannot be empty.")

        return self.retriever.retrieve(query)
