"""
Compiles the LangGraph application. app.py imports build_graph() and
calls .stream() on it — the agent knows nothing about Streamlit.
"""

from langgraph.graph import StateGraph, START, END

from .state import ResearchState
from .nodes import planner, researcher, critic, decision, reporter
from .router import route


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("critic", critic)
    builder.add_node("decision", decision)
    builder.add_node("reporter", reporter)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "critic")
    builder.add_edge("critic", "decision")
    # The real feedback loop: on retry, control goes back to the Planner
    # (not straight to Researcher), so the next pass re-plans using the
    # Critic's critique rather than just re-running research on old tasks.
    builder.add_conditional_edges("decision", route, {"retry": "planner", "approve": "reporter"})
    builder.add_edge("reporter", END)

    return builder.compile()