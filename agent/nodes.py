"""
The five agent nodes: Planner, Researcher, Critic, Decision, Reporter.

Each LLM call goes through _invoke_llm() so token usage is tracked centrally
(app.py reads get_token_usage() after each graph.stream() event to drive the
sidebar's running token counter).

Every node works with or without a GROQ_API_KEY: no key -> deterministic
mock behaviour, so the retry loop is always demonstrable even offline
"""

import os
import re
import json

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from .rag import retrieve

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2

_token_usage = {"total": 0}


def reset_token_usage() -> None:
    _token_usage["total"] = 0


def get_token_usage() -> int:
    return _token_usage["total"]


class FinalReport(BaseModel):
    """A a Pydantic-validated structured report."""
    goal: str
    summary: str
    key_findings: list[str] = Field(min_length=1)
    risks: list[str]
    sources: list[str]
    iterations: int
    confidence: str = Field(description="high | medium | low")


def _get_llm(config):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    cfg = (config or {}).get("configurable", {}) or {}
    model = cfg.get("model", DEFAULT_MODEL)
    temperature = cfg.get("temperature", DEFAULT_TEMPERATURE)
    return ChatGroq(model=model, temperature=temperature)


async def _invoke_llm(llm, prompt: str) -> str:
    resp = await llm.ainvoke(prompt)
    usage = getattr(resp, "usage_metadata", None) or {}
    tokens = usage.get("total_tokens", 0) if usage else 0
    _token_usage["total"] += tokens
    return resp.content


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# Planner:
# decomposes the goal into tasks. On retry it reads the Critic's critique from state and re-plans around it, 
# instead of repeating the same plan

async def planner(state, config=None):
    goal = state["goal"]
    critique = state.get("critique", "")
    llm = _get_llm(config)

    if llm:
        prompt = (
            f"You are a planning agent. Goal: {goal}.\n"
            f"Previous critique to address (empty on first pass): {critique or 'none'}.\n"
            "Return exactly 3 short, concrete research task bullets that specifically "
            "address any gaps named above. One per line, no numbering, no extra text."
        )
        text = await _invoke_llm(llm, prompt)
        tasks = [t.strip("-* ").strip() for t in text.splitlines() if t.strip()][:3]
        if not tasks:
            tasks = [f"Research: {goal}"]
    else:
        base = [f"Define the scope of '{goal}'", "Gather key facts", "Identify risks and gaps"]
        tasks = base if not critique else base + [f"Address gap: {critique[:60]}"]

    print(f"[planner] retry_count={state.get('retry_count', 0)} -> {len(tasks)} tasks"
          + (f'  [addressing: "{critique[:40]}..."]' if critique else ""))
    return {"tasks": tasks}


# Researcher:
# grounds every task in the uploaded/indexed documents via RAG.
# Sources are embedded in each finding string (e.g. [Sources: doc.pdf])
# so the Reporter can recover them without a non-contract state key.

async def researcher(state, config=None):
    llm = _get_llm(config)
    tasks = state["tasks"]
    findings = []

    for task in tasks:
        context, sources = retrieve(task)

        if not context:
            findings.append("[Sources: none] No indexed documents cover: " + task)
            continue

        if llm:
            prompt = (
                "Answer ONLY from the context below. Be concise (2-3 sentences). "
                "If the context does not actually cover the task, say so explicitly "
                "instead of guessing.\n\n"
                f"Task: {task}\n\nContext:\n{context}"
            )
            text = await _invoke_llm(llm, prompt)
        else:
            text = f"[mock finding] {task}: " + context[:200]

        tag = f"[Sources: {', '.join(sources)}] " if sources else "[Sources: none] "
        findings.append(tag + text)

    print(f"[researcher] {len(findings)} findings gathered")
    return {"findings": findings}


# Critic:
# scores completeness 0-1 and names concrete gaps. The gaps (not
# just the score) are what let the Planner meaningfully re-plan.

async def critic(state, config=None):
    llm = _get_llm(config)
    goal = state["goal"]
    findings_text = "\n".join(state.get("findings", []))

    if llm:
        prompt = (
            "You are a strict reviewer. Score the findings for completeness vs the goal.\n"
            'Return ONLY JSON: {"score": <0..1 float>, "gaps": "<one concise sentence>"}.\n'
            f"Goal: {goal}\nFindings: {findings_text}"
        )
        raw = await _invoke_llm(llm, prompt)
        data = _extract_json(raw)
        try:
            score = float(data.get("score", 0.5))
            gaps = str(data.get("gaps", ""))
        except (TypeError, ValueError):
            score, gaps = 0.5, "Could not parse critic output; treating as incomplete."
    else:
        no_doc_hits = "no indexed documents" in findings_text.lower()
        has_risk_mention = "risk" in findings_text.lower()
        if no_doc_hits:
            score, gaps = 0.3, "Findings could not be grounded in any indexed document."
        elif state.get("retry_count", 0) == 0:
            # Deliberately weak on the first pass so the retry loop is always
            # demonstrable in mock mode (task brief explicitly allows/expects
            # forcing a low first-pass score for the demo).
            score, gaps = 0.6, "Findings omit key risks; add security, regulation, over-reliance."
        else:
            score, gaps = 0.9, ""

    print(f"[critic] quality_score={round(score, 2)}" + (f" | gap: {gaps[:60]}" if gaps else ""))
    return {"quality_score": score, "critique": gaps}


# Decision:
# logs the routing choice AND is the single place retry_count is incremented, 
# so router.route() only ever reads state (no side effects in
# the conditional-edge function itself).

async def decision(state, config=None):
    from .router import THRESHOLD, MAX_RETRIES

    score = state["quality_score"]
    retry_count = state.get("retry_count", 0)

    if score >= THRESHOLD:
        print(f"[decision] score {round(score, 2)} >= {THRESHOLD} -> approve")
        return {}

    if retry_count >= MAX_RETRIES:
        print(f"[decision] score {round(score, 2)} < {THRESHOLD} but retry cap "
              f"({MAX_RETRIES}) reached -> approve (best-effort, below threshold)")
        return {}

    new_count = retry_count + 1
    print(f"[decision] score {round(score, 2)} < {THRESHOLD} -> retry (attempt {new_count})")
    return {"retry_count": new_count}


# Reporter:
# assembles and VALIDATES the final report via Pydantic. 
# Sources are recovered from the "[Sources: ...]" tags Researcher embedded.

_SOURCE_TAG = re.compile(r"^\[Sources: ([^\]]*)\]\s*")


async def reporter(state, config=None):
    goal = state["goal"]
    findings = state.get("findings", [])
    findings_text = "\n".join(findings)
    risks = [f for f in findings if "risk" in f.lower()] or ["No explicit risks surfaced; see findings."]

    all_sources = set()
    for f in findings:
        m = _SOURCE_TAG.match(f)
        if m and m.group(1) != "none":
            all_sources.update(s.strip() for s in m.group(1).split(",") if s.strip())

    score = state["quality_score"]
    confidence = "high" if score >= 0.8 else ("medium" if score >= 0.5 else "low")

    report = FinalReport(
        goal=goal,
        summary=(findings_text[:400] if findings_text else "No findings gathered."),
        key_findings=findings or [f"No findings gathered for goal: {goal}"],
        risks=risks,
        sources=sorted(all_sources),
        iterations=state.get("retry_count", 0) + 1,
        confidence=confidence,
    )
    print("[reporter] validated FinalReport produced")
    return {"report": report.model_dump()}
