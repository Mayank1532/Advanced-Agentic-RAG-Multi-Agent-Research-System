import time
from pathlib import Path

from agentic_rag.rag import LocalQwenQueryRewriter


MODEL_ROOT = Path(
    r"D:\HuggingFaceCache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots"
)


def test_real_qwen_query_rewriting() -> None:
    model_path = next(MODEL_ROOT.iterdir())

    rewriter = LocalQwenQueryRewriter(model_path)

    query = (
        "How can a RAG system retrieve useful information "
        "when the user's wording differs from the source documents?"
    )

    start = time.perf_counter()

    rewritten = rewriter.rewrite(query)

    elapsed = time.perf_counter() - start

    print(f"\nOriginal: {query}")
    print(f"Rewritten: {rewritten}")
    print(f"Rewrite latency: {elapsed:.2f}s")

    assert rewritten
    assert len(rewritten) > 10
