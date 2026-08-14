from agentic_rag.graph import build_research_graph


def test_research_graph_runs_end_to_end() -> None:
    graph = build_research_graph()

    result = graph.invoke(
        {
            "question": "How does hybrid retrieval improve RAG?",
        }
    )

    assert result["retrieval_query"] == (
        "How does hybrid retrieval improve RAG?"
    )
    assert result["evidence"]
    assert result["analysis"]
    assert result["draft"]
    assert result["review"]
    assert result["approved"] is True
