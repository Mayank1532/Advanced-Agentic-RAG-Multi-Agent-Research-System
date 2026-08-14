from agentic_rag.agents import (
    AnalystAgent,
    ResearcherAgent,
    ReviewerAgent,
    WriterAgent,
)
from agentic_rag.graph.researcher import ResearchRetriever
from agentic_rag.graph.state import ResearchState


_retriever: ResearchRetriever | None = None
_researcher: ResearcherAgent | None = None
_analyst = AnalystAgent()
_writer = WriterAgent()
_reviewer = ReviewerAgent()


def planner_node(state: ResearchState) -> ResearchState:
    """Plan the research task."""

    question = state["question"].strip()

    if not question:
        raise ValueError("Research question cannot be empty.")

    return {
        "retrieval_query": question,
        "revision_count": state.get("revision_count", 0),
    }


def researcher_node(state: ResearchState) -> ResearchState:
    """Gather evidence using the Advanced RAG pipeline."""

    global _retriever, _researcher

    if _retriever is None:
        _retriever = ResearchRetriever()

    if _researcher is None:
        _researcher = ResearcherAgent(_retriever)

    evidence = _researcher.research(
        state["retrieval_query"],
    )

    return {"evidence": evidence}


def analyst_node(state: ResearchState) -> ResearchState:
    """Analyze and synthesize retrieved evidence."""

    return {
        "analysis": _analyst.analyze(
            state.get("evidence", []),
        ),
    }


def writer_node(state: ResearchState) -> ResearchState:
    """Write the research report."""

    return {
        "draft": _writer.write(
            state["question"],
            state.get("analysis", ""),
        ),
    }


def reviewer_node(state: ResearchState) -> ResearchState:
    """Review the generated report against retrieved evidence."""

    review, approved = _reviewer.review(
        state.get("draft", ""),
        state.get("evidence", []),
    )

    return {
        "review": review,
        "approved": approved,
    }


def review_router(state: ResearchState) -> str:
    """Route approved reports to END and rejected reports to revision."""

    if state.get("approved", False):
        return "approved"

    if state.get("revision_count", 0) >= 1:
        return "approved"

    return "revision"
