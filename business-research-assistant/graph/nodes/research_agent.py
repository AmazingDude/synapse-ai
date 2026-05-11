"""
Research Agent — second node in the business research pipeline.

Runs three targeted Tavily searches and asks the LLM to synthesise the raw
results into a structured four-section report with a self-assigned confidence
score (0-10).
"""

import json
import logging  # CHANGED: replaced print() with logging
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from graph.state import GraphState
from tools.search import format_search_results, web_search
from utils.llm import get_llm  # CHANGED: lazy singleton
from utils.parsing import extract_outermost_json  # CHANGED: shared parser

logger = logging.getLogger(__name__)  # CHANGED

# Common question/command prefixes stripped before building search queries.
_STRIP_PREFIX_RE = re.compile(
    r"^(what\s+(is|are|was|were)\s+(the\s+)?|tell\s+me\s+about\s+|"
    r"give\s+me\s+(a\s+)?(summary\s+of\s+|overview\s+of\s+)?|"
    r"can\s+you\s+(research\s+|find\s+)?|find\s+(out\s+)?about\s+|"
    r"how\s+(is|are|does|do)\s+|describe\s+|research\s+|analyze\s+)",
    re.IGNORECASE,
)

# Collapses any whitespace run (incl. newlines from combined multi-turn queries) to a single space.
_WHITESPACE_RE = re.compile(r"\s+")  # CHANGED: handle combined_query newlines

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior business research analyst. Your task is to synthesise raw
web search results into a concise, factual business report.

Structure your report with these four sections (use the exact headers shown):

## Company Overview
A brief description of the company / entity, its business model, sector, and scale.

## Recent News
Key events, announcements, or developments from the past 12-18 months.

## Financial Information
Revenue, profit, funding rounds, valuations, or any quantitative metrics found.
Write "Data not found" if no reliable numbers are available.

## Key Developments
Product launches, leadership changes, partnerships, regulatory events, or
strategic pivots that matter to a business analyst.

Rules:
- Extract specific facts, numbers, named sources, and dates wherever present.
- If a section has no reliable data, write "Data not found" under that heading.
- Flag unverified or contradictory claims with "(unverified)".
- Do NOT invent or extrapolate information not present in the search results.

After writing the report, assign a confidence_score (float 0.0-10.0):
  0-2  : Almost no usable data; mostly search errors or completely off-topic results.
  3-5  : Partial coverage; at least one section is well-supported but major gaps remain.
  6-8  : Good coverage; most sections have concrete data, only minor gaps.
  9-10 : Comprehensive; all four sections are well-sourced with specific facts.

Respond with ONLY a single valid JSON object (no markdown fences):
{
  "findings": "<your full markdown report as a single escaped string>",
  "confidence_score": <float>
}
"""


def _extract_subject(query: str) -> str:
    """Derive a clean search subject from the (possibly multi-line) user query."""
    one_line = _WHITESPACE_RE.sub(" ", query).strip()  # CHANGED: collapse newlines from combined_query
    stripped = _STRIP_PREFIX_RE.sub("", one_line).strip(" ?.,!").strip()  # CHANGED
    return stripped or one_line or query.strip()  # CHANGED


def _build_search_queries(subject: str) -> list[str]:
    return [
        f"{subject} latest news 2025 2026",
        f"{subject} financials revenue business overview",
        f"{subject} recent developments products leadership",
    ]


def _run_searches(queries: list[str]) -> tuple[str, int]:
    """Execute every search; return (aggregated_text, success_count)."""
    chunks: list[str] = []
    success_count = 0

    for q in queries:
        try:
            resp = web_search(q, max_results=5)
            formatted = format_search_results(resp)
            chunks.append(f"### Search: {q}\n{formatted}")
            success_count += 1
            logger.info(  # CHANGED: was print()
                "Search OK: %r (%d results)",
                q,
                len(resp.get("results", [])),
            )
        except Exception as exc:  # noqa: BLE001
            chunks.append(f"### Search: {q}\n(search error: {exc})")
            logger.error("Search FAILED: %r - %s", q, exc)  # CHANGED: was print()

    return "\n\n".join(chunks), success_count


def _parse_synthesis(raw: str, success_count: int) -> tuple[str, float]:
    """Parse the LLM JSON synthesis; fall back gracefully on errors."""
    json_str = extract_outermost_json(raw)  # CHANGED: shared helper
    if json_str:
        try:
            data = json.loads(json_str)
            findings = str(data.get("findings", "")).strip() or raw.strip()
            score = float(data.get("confidence_score", 0.0))
            score = max(0.0, min(10.0, score))
            return findings, score
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("JSON parse error in synthesis response: %s", exc)  # CHANGED

    fallback_score = min(6.0, float(success_count) * 2.0)
    logger.warning("Using fallback synthesis output: confidence=%.1f", fallback_score)  # CHANGED
    return raw.strip(), fallback_score


def _recent_message_context(messages: list[BaseMessage], n: int = 4) -> str:
    """Compact recent-conversation context for the synthesis prompt."""
    lines: list[str] = []
    for msg in messages[-n:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {str(msg.content).strip()}")
    return "\n".join(lines)


def research_agent(state: GraphState) -> dict:
    """Run targeted Tavily searches and synthesise structured business findings."""
    query = (state.get("original_query") or "").strip()
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content).strip()
                break

    subject = _extract_subject(query)
    logger.info("Subject: %r", subject)  # CHANGED

    search_queries = _build_search_queries(subject)
    aggregated_results, success_count = _run_searches(search_queries)
    logger.info(  # CHANGED
        "%d/%d searches succeeded", success_count, len(search_queries),
    )

    context_snippet = _recent_message_context(state["messages"])

    synthesis_prompt = [
        SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original research query:\n{query}\n\n"
                f"Recent conversation context:\n{context_snippet}\n\n"
                f"Raw search results:\n{aggregated_results}"
            )
        ),
    ]

    response = get_llm().invoke(synthesis_prompt)  # CHANGED: lazy llm
    raw_text = str(response.content) if hasattr(response, "content") else str(response)

    findings, confidence = _parse_synthesis(raw_text, success_count)
    logger.info("confidence_score=%.1f", confidence)  # CHANGED

    return {
        "research_findings": findings,
        "confidence_score": confidence,
        "research_attempts": (state.get("research_attempts") or 0) + 1,
    }
