from langgraph.graph import END, START, StateGraph

from agentic_rag.graph.nodes import (
    analyst_node,
    planner_node,
    researcher_node,
    review_router,
    reviewer_node,
    writer_node,
)
from agentic_rag.graph.state import ResearchState


def build_research_graph():
    """Build the Project 7 research workflow."""

    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "approved": END,
            "revision": "writer",
        },
    )

    return graph.compile()
