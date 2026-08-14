from agentic_rag.graph.state import ResearchState


def build_research_graph():
    """Build the LangGraph research workflow lazily."""
    from agentic_rag.graph.workflow import build_research_graph as _build

    return _build()


__all__ = [
    "ResearchState",
    "build_research_graph",
]
