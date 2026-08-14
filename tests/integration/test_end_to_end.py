from agentic_rag.graph import build_research_graph


def test_complete_research_workflow() -> None:
    graph = build_research_graph()

    result = graph.invoke(
        {
            "question": (
                "How does hybrid retrieval improve "
                "the quality of RAG systems?"
            )
        }
    )

    assert result["retrieval_query"]
    assert len(result["evidence"]) >= 1
    assert result["analysis"]
    assert result["draft"]
    assert result["review"]
    assert result["approved"] is True

    assert "# Research Report" in result["draft"]
    assert "## Question" in result["draft"]
    assert "## Analysis" in result["draft"]
