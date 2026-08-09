from typing import TypedDict


class ResearchState(TypedDict):
    goal: str                  # the user's research objective
    tasks: list[str]           # produced by the Planner
    findings: list[str]        # produced by the Researcher
    critique: str               # the Critic's written feedback / gaps
    quality_score: float        # 0.0 - 1.0, produced by the Critic
    retry_count: int            # incremented on every loop back
    report: dict                 # final structured output (validated Pydantic model, dumped to dict)