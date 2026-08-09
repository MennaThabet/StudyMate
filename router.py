"""
Router — the conditional edge after the Decision node.

The constants here are deliberately named and centralized (not buried inside a function)
app.py imports these same values to display them in the sidebar rather than hardcoding a second copy.
"""

THRESHOLD = 0.8    # quality bar a report must clear to be approved
MAX_RETRIES = 4     # safety cap: after this many retries, approve best-effort and stop


def route(state) -> str:
    """Reads state only. The actual retry_count increment happens inside
    the `decision` node (agent/nodes.py) — this function is a pure router."""
    if state["quality_score"] >= THRESHOLD:
        return "approve"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "approve"
    return "retry"