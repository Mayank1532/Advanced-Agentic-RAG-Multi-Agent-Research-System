from agentic_rag.agents import (
    AnalystAgent,
    ReviewerAgent,
    WriterAgent,
)


def test_analyst_synthesizes_evidence() -> None:
    result = AnalystAgent().analyze(
        ["Hybrid search combines vector and keyword retrieval."]
    )

    assert "Evidence 1:" in result
    assert "Hybrid search" in result


def test_writer_creates_report() -> None:
    result = WriterAgent().write(
        "What is hybrid retrieval?",
        "Hybrid retrieval combines complementary retrieval methods.",
    )

    assert "# Research Report" in result
    assert "What is hybrid retrieval?" in result


def test_reviewer_approves_evidence_grounded_report() -> None:
    draft = (
        "# Research Report\n\n"
        "## Question\n\n"
        "What is hybrid retrieval?\n\n"
        "## Analysis\n\n"
        "Hybrid retrieval combines retrieval methods."
    )

    evidence = [
        "Hybrid retrieval combines retrieval methods.",
    ]

    review, approved = ReviewerAgent().review(
        draft,
        evidence,
    )

    assert approved is True
    assert "Approved" in review


def test_reviewer_rejects_missing_evidence() -> None:
    draft = (
        "# Research Report\n\n"
        "## Question\n\n"
        "What is hybrid retrieval?\n\n"
        "## Analysis\n\n"
        "Some claim."
    )

    _, approved = ReviewerAgent().review(
        draft,
        [],
    )

    assert approved is False
