# StudyMate Research Agent — Multi-Agent Dashboard

An autonomous LangGraph agent (Planner → Researcher → Critic → Decision → Reporter)
with a Streamlit dashboard that runs it and streams its reasoning live, including
watching it fail its own quality check, retry itself, and improve — with no human
in the loop.

## Demo

**`StudyMate Research Agent Demo.mp4`** (repo root) shows a full run end to end:
an objective entered → the reasoning stream (Planner → Researcher → Critic →
Decision) → a failed quality check → an automatic retry → approval → the final
report rendered and exported. Clone the repo and open the file directly, or
watch it inline if your Git host renders `.mp4` previews.

A written run is also included at **`agent_report.md`** (repo root) — a sample
Markdown export produced by the Reporter node, generated the same way the
in-app download button produces one.

## Architecture

```
agent/
├── .gitkeep         # keeps the package directory tracked before other files existed
├── __init__.py       # marks agent/ as a Python package
├── state.py          # ResearchState (TypedDict) — the shared state schema
├── nodes.py           # planner, researcher, critic, decision, reporter
├── router.py         # conditional-edge routing logic + THRESHOLD / MAX_RETRIES
├── rag.py             # document indexing + retrieval for the Researcher
└── graph.py           # build_graph() -> compiled LangGraph app
app.py                 # Streamlit dashboard — imports and runs the agent, in-process
.env                    # your real API key (gitignored; not committed)
.env.example            # template showing the expected environment variables
requirements.txt        # pinned dependencies
agent_report.md          # sample Markdown report exported from a real run
StudyMate Research Agent Demo.mp4   # full walkthrough recording (see Demo above)
README.md
```

The agent knows nothing about Streamlit — `app.py` only imports `build_graph()`
and calls `.stream()` on it. This keeps the seam clean for Session 5, where the
same agent moves behind a FastAPI service with no rewrite.

## Setup (local)

1. Clone the repo and `cd` into it.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\Activate.ps1        # Windows PowerShell
   # source venv/bin/activate       # Mac/Linux
   ```
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your real Groq API key:
   ```bash
   GROQ_API_KEY=gsk_...
   ```
5. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
6. Open the URL Streamlit prints (usually `http://localhost:8501`).

The dashboard also runs without a key — leave `.env` empty and it falls back to
deterministic mock agents, so the retry loop is always demonstrable offline.

## Setup (Colab + tunnel)

1. Upload this repo to a Colab notebook (or `git clone` it inside Colab).
2. Install dependencies: `!pip install -r requirements.txt`
3. Install a tunnel tool, e.g. `pyngrok`:
   ```python
   !pip install pyngrok
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
   ```
4. Launch Streamlit in the background and open the tunnel:
   ```python
   !streamlit run app.py &>/content/log.txt &
   public_url = ngrok.connect(8501)
   print(public_url)
   ```
5. Open the printed public URL.

## How to use it

1. (Optional) Upload course documents in the sidebar and click **Index documents**
   — this routes them into the Researcher's knowledge source (see `agent/rag.py`).
2. Pick a model and temperature in the sidebar.
3. Type a research objective into the chat box.
4. Watch the reasoning stream: Planner → Researcher → Critic → Decision, live.
5. If the Critic scores below **0.8**, the Decision node loops back to the
   Planner (which re-plans using the critique) — you'll see this happen live,
   up to **2** retries, after which it approves best-effort.
6. Once approved, the final report renders inline and can be exported as
   Markdown via the download button (see `agent_report.md` for a sample export).

## Guardrails

- **Quality threshold:** `0.8` (named constant in `agent/router.py`, shown in the sidebar)
- **Retry cap:** `MAX_RETRIES = 2` (same file) — the graph always terminates,
  approving best-effort with a lower confidence flag if the cap is hit.

## Bonus features implemented

- **Live graph visualization** highlighting the currently-executing node (Mermaid, rendered inline).
- **Per-cycle history panel** — score, decision, and critique for every retry, shown side by side.
- **Cost estimate** next to the token counter, using an approximate per-model $/1M-token table.

## Known limitations

*(To be filled in honestly once you've run it a few times — e.g. how the
Researcher handles an empty knowledge base by explicitly reporting "no
documents cover this," rather than inventing findings.)*