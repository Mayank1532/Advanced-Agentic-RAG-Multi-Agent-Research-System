from agentic_rag.rag.query_rewriter import QueryRewriter


class FakeQueryRewriter:
    def rewrite(self, query: str) -> str:
        return f"rewritten: {query}"


def test_query_rewriter_contract() -> None:
    rewriter: QueryRewriter = FakeQueryRewriter()

    assert rewriter.rewrite("What is hybrid search?") == (
        "rewritten: What is hybrid search?"
    )


def test_query_rewriter_preserves_empty_result_contract() -> None:
    rewriter: QueryRewriter = FakeQueryRewriter()

    result = rewriter.rewrite("")

    assert result == "rewritten: "
