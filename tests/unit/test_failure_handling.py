import pytest

from agentic_rag.agents import AnalystAgent, ReviewerAgent, WriterAgent
from agentic_rag.graph import build_research_graph


def test_planner_rejects_empty_question() -> None:
    graph = build_research_graph()

    with pytest.raises(ValueError, match="Research question cannot be empty"):
        graph.invoke({"question": ""})


def test_analyst_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError, match="empty evidence"):
        AnalystAgent().analyze([])


def test_writer_rejects_empty_analysis() -> None:
    with pytest.raises(ValueError, match="Analysis cannot be empty"):
        WriterAgent().write(
            "What is hybrid retrieval?",
            "",
        )


def test_reviewer_rejects_empty_draft() -> None:
    review, approved = ReviewerAgent().review(
        "",
        ["Some evidence."],
    )

    assert approved is False
    assert "empty" in review.lower()


def test_reviewer_rejects_missing_evidence() -> None:
    review, approved = ReviewerAgent().review(
        (
            "# Research Report\n\n"
            "## Question\n\n"
            "What is RAG?\n\n"
            "## Analysis\n\n"
            "Some analysis."
        ),
        [],
    )

    assert approved is False
    assert "evidence" in review.lower()
