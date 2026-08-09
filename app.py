"""
Streamlit dashboard — imports and runs the compiled agent graph in-process.

UI calls agent; agent knows nothing about Streamlit.
All agent logic lives in agent/*.py — nothing here reimplements
planning, research, critique, or reporting.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; if not installed, rely on system env vars instead

from agent.graph import build_graph
from agent.state import ResearchState
from agent.router import THRESHOLD, MAX_RETRIES
from agent.nodes import reset_token_usage, get_token_usage
from agent.rag import index_documents

# Rough, approximate per-1M-token USD pricing for cost estimate.
# These are illustrative, not billing-accurate — check Groq's current pricing
# page before treating this as a real cost tracker.
MODEL_PRICES_PER_1M = {
    "llama-3.3-70b-versatile": 0.59,
    "llama-3.1-8b-instant": 0.05,
    "openai/gpt-oss-20b": 0.10,
}

NODE_ICONS = {"planner": "🧭", "researcher": "🔎", "critic": "🧐", "decision": "⚖️", "reporter": "📝"}

st.set_page_config(page_title="StudyMate Research Agent", layout="wide")

# ---------------------------------------------------------------------------
# session_state: survives Streamlit's top-to-bottom re-runs
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tokens" not in st.session_state:
    st.session_state.tokens = 0
if "last_report_md" not in st.session_state:
    st.session_state.last_report_md = ""
if "cycle_history" not in st.session_state:
    st.session_state.cycle_history = []  # list of {iteration, score, decision, critique}
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0

graph_app = build_graph()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def render_report_markdown(report: dict) -> str:
    findings = "\n".join(f"- {f}" for f in report["key_findings"])
    risks = "\n".join(f"- {r}" for r in report["risks"])
    sources = ", ".join(report["sources"]) if report["sources"] else "_None_"
    return f"""# Research Report: {report['goal']}

**Confidence:** {report['confidence']}  |  **Iterations:** {report['iterations']}

## Summary
{report['summary']}

## Key Findings
{findings}

## Risks
{risks}

## Sources
{sources}
"""


def mermaid_html(active_node: str | None) -> str:
    """Renders the graph as Mermaid, highlighting the currently-executing
    node ('visual graph... highlighting the node currently executing'). 
    Uses the mermaid.js CDN directly since Streamlit has no native Mermaid support."""
    
    nodes = ["planner", "researcher", "critic", "decision", "reporter"]
    style = ""
    if active_node in nodes:
        style = f"style {active_node} fill:#ffd166,stroke:#333,stroke-width:3px"
    diagram = f"""
    graph LR
      planner[🧭 Planner] --> researcher[🔎 Researcher]
      researcher --> critic[🧐 Critic]
      critic --> decision{{⚖️ Decision}}
      decision -->|retry| planner
      decision -->|approve| reporter[📝 Reporter]
      {style}
    """
    return f"""
    <div class="mermaid">{diagram}</div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true }});
    </script>
    """


# ---------------------------------------------------------------------------
# sidebar: key + controls + file upload + token counter + cost estimate
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    default_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API key", value=default_key, type="password",
        help="Auto-loaded from .env (GROQ_API_KEY). Blank = mock mode.",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    st.caption("🟢 Groq LLM: ON" if api_key else "⚪ Mock mode (no key)")

    model = st.selectbox("Model", list(MODEL_PRICES_PER_1M.keys()))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    st.caption(f"Quality threshold: **{THRESHOLD}**  |  Max retries: **{MAX_RETRIES}**")

    st.divider()
    uploaded_files = st.file_uploader(
        "Course documents (PDF / TXT / MD)", type=["pdf", "txt", "md"], accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index documents"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for f in uploaded_files:
                p = Path(tmp_dir) / f.name
                p.write_bytes(f.getvalue())
                paths.append(str(p))
            with st.spinner("Indexing documents..."):
                st.session_state.indexed_chunks = index_documents(paths)
        st.success(f"Indexed {st.session_state.indexed_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.indexed_chunks:
        st.caption(f"📚 Knowledge base: {st.session_state.indexed_chunks} chunks indexed")
    else:
        st.caption("📚 No documents indexed yet")

    st.divider()
    st.metric("Tokens used (session)", st.session_state.tokens)
    est_cost = st.session_state.tokens / 1_000_000 * MODEL_PRICES_PER_1M.get(model, 0.0)
    st.caption(f"💵 Estimated cost so far: ${est_cost:.4f} (at current model's rate)")

    if st.button("Reset session"):
        st.session_state.messages = []
        st.session_state.tokens = 0
        st.session_state.cycle_history = []
        st.session_state.last_report_md = ""
        st.rerun()

st.title("🤖 StudyMate Research Agent")
st.caption("Plans → researches (RAG-grounded) → critiques itself → retries if below threshold → reports")

# ---------------------------------------------------------------------------
# replay chat history
# ---------------------------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------------------------------------------------------------------
# chat input -> run the compiled graph -> stream reasoning live
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Enter a research objective..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        graph_placeholder = st.empty()
        state_placeholder = st.empty()

        initial_state = ResearchState(
            goal=prompt, tasks=[], findings=[], critique="",
            quality_score=0.0, retry_count=0, report={},
        )
        config = {"configurable": {"model": model, "temperature": temperature}}

        reset_token_usage()
        last_seen_tokens = 0
        final_report = {}
        state_acc = dict(initial_state)  # .stream() yields per-node deltas only; accumulate manually

        with st.status("Agent working...", expanded=True) as status:
            try:
                for event in graph_app.stream(initial_state, config=config):
                    for node_name, update in event.items():
                        if isinstance(update, dict):
                            state_acc.update(update)

                        current_tokens = get_token_usage()
                        delta = current_tokens - last_seen_tokens
                        last_seen_tokens = current_tokens
                        st.session_state.tokens += delta

                        graph_placeholder.empty()
                        with graph_placeholder.container():
                            st.components.v1.html(mermaid_html(node_name), height=180)

                        icon = NODE_ICONS.get(node_name, "•")
                        st.write(f"{icon} **{node_name}** fired")

                        if node_name == "decision":
                            # snapshot the cycle for the history panel — read from the
                            # accumulated state, since critique/score were set by the Critic
                            # node just before this one, not by decision's own (small) update.
                            outcome = "retry" if isinstance(update, dict) and "retry_count" in update else "approve"
                            st.session_state.cycle_history.append({
                                "iteration": len(st.session_state.cycle_history) + 1,
                                "score": round(state_acc.get("quality_score", 0.0), 2),
                                "decision": outcome,
                                "critique": state_acc.get("critique", "") or "—",
                            })

                        if node_name == "reporter":
                            final_report = update["report"]

                status.update(label="Done ✅", state="complete")
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Agent failed: {e}\n\nCheck your Groq key/model, or clear the key to use mock mode.")

        if final_report:
            report_md = render_report_markdown(final_report)
            st.markdown("### Final Report")
            st.markdown(report_md)
            st.session_state.last_report_md = report_md
            st.session_state.messages.append({"role": "assistant", "content": report_md})

            with st.expander("📊 State view (final)"):
                st.json({
                    "confidence": final_report["confidence"],
                    "iterations": final_report["iterations"],
                    "sources": final_report["sources"],
                })

# ---------------------------------------------------------------------------
# per-cycle history panel
# ---------------------------------------------------------------------------
if st.session_state.cycle_history:
    with st.expander("🔁 Retry history (this run)"):
        st.table(st.session_state.cycle_history)

# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
if st.session_state.last_report_md:
    st.download_button(
        "⬇️ Export report (Markdown)",
        data=st.session_state.last_report_md,
        file_name="agent_report.md",
        mime="text/markdown",
    )