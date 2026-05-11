"""
Synthesis Agent — fourth and final node in the business research pipeline.

Transforms validated research findings into a polished user-facing report.

Report structure (enforced via system prompt)
---------------------------------------------
    ## 🏢 Company Overview
    ## 📰 Recent News & Developments
    ## 💰 Financial Snapshot
    ## 👥 Leadership & Strategy
    ## 🔑 Key Takeaways

The report always ends with a single-line "Sources note" disclaimer.  When
``confidence_score < 6``, a visible warning block is added at the top.  The
last 8 messages of conversation history are included so multi-turn follow-ups
("what about their competitors?") are answered with proper context.
"""

import logging  # CHANGED: replaced print() with logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from graph.state import GraphState
from utils.llm import get_llm  # CHANGED: lazy singleton

logger = logging.getLogger(__name__)  # CHANGED

# Confidence < this triggers the visible low-confidence warning at the top.
_LOW_CONFIDENCE_THRESHOLD = 6.0  # CHANGED: was 5.0; aligned to the user-requested rule

# How many recent messages to forward as conversation context.
_CONTEXT_MESSAGE_COUNT = 8  # CHANGED: was 6; user requested 8

# CHANGED: full prompt rewrite — emoji headers, low-conf disclaimer rule,
#          mandatory Sources note, multi-turn awareness.
_SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior business research analyst writing a structured report for a
professional audience.  Your tone is clear, direct, and professional - like a
well-written briefing note.

USE EXACTLY THESE FIVE SECTION HEADERS IN THIS ORDER
----------------------------------------------------
## 🏢 Company Overview
A concise description of the company or subject: what it does, its sector,
approximate size, and market position.

## 📰 Recent News & Developments
Key events, product launches, partnerships, or strategic moves from the past
12-18 months.  Include dates and names where available.

## 💰 Financial Snapshot
Revenue, profit, valuation, funding rounds, or any financial metrics found.
If no financial data is available, write:
> Financial data was not available in the research results.

## 👥 Leadership & Strategy
Key executives (named where possible), strategic direction, organisational
changes, and any leadership commentary that surfaced in the research.

## 🔑 Key Takeaways
3-5 concise bullet points capturing the most important conclusions a
decision-maker needs to know.

LOW-CONFIDENCE DISCLAIMER
-------------------------
If the inputs include "Low confidence flag: true", you MUST begin the report
(BEFORE the first ## header) with EXACTLY this blockquote line:

> ⚠️ Data confidence is low. Some information may be incomplete or outdated.

If "Low confidence flag: false", omit the disclaimer.

MANDATORY CLOSING NOTE
----------------------
Always end the report with this exact single line:

_Sources note: information above is drawn from public web sources and may not reflect real-time market data._

RULES
-----
- Only state facts supported by the research findings provided.
- Do not fabricate figures, dates, or events.
- If a section genuinely has no data, say so clearly rather than padding.
- Use the conversation history to handle follow-up questions correctly
  (e.g. when the user says "what about their competitors", "they" refers to
  the company discussed earlier in the conversation).
- Write in flowing prose for the first four sections; bullets only in
  Key Takeaways.
"""


def _is_low_confidence(state: GraphState) -> bool:  # CHANGED: renamed from _is_low_quality
    """Return True when the low-confidence disclaimer should be added."""
    score = state.get("confidence_score")
    if score is not None and score < _LOW_CONFIDENCE_THRESHOLD:
        return True
    if state.get("validation_result") == "insufficient":
        return True
    return False


def _format_conversation_context(messages: list[BaseMessage], n: int) -> str:
    """Last *n* messages as 'User: ...' / 'Assistant: ...' lines."""
    recent = messages[-n:] if len(messages) > n else messages[:]
    lines: list[str] = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "System"
        content = str(msg.content).strip()
        if len(content) > 400:
            content = content[:400] + "... [truncated]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_human_message(state: GraphState, low_confidence: bool) -> str:  # CHANGED: pass flag explicitly
    """Compose the human-turn payload carrying all data to the LLM."""
    query = (state.get("original_query") or "").strip()
    validation = state.get("validation_result") or "not assessed"
    confidence = state.get("confidence_score")
    findings = (state.get("research_findings") or "").strip()

    confidence_str = f"{confidence:.1f}/10" if confidence is not None else "not available"

    context_block = _format_conversation_context(
        state["messages"], _CONTEXT_MESSAGE_COUNT,
    )

    parts = [
        f"Original user query:\n{query}",
        f"Validation result: {validation}",
        f"Research confidence score: {confidence_str}",
        f"Low confidence flag: {'true' if low_confidence else 'false'}",  # CHANGED: explicit flag the prompt reads
    ]

    if context_block:
        parts.append(
            f"Conversation history (last {_CONTEXT_MESSAGE_COUNT} turns):\n{context_block}",
        )

    if findings:
        parts.append(f"Research findings:\n{findings}")
    else:
        parts.append("Research findings: No findings were retrieved.")

    return "\n\n".join(parts)


def synthesis_agent(state: GraphState) -> dict:
    """Generate the final user-facing markdown report."""
    low_confidence = _is_low_confidence(state)  # CHANGED: renamed helper

    if low_confidence:
        logger.info(  # CHANGED: was print()
            "Low-confidence signal (validation=%r, confidence=%s); disclaimer requested.",
            state.get("validation_result"),
            state.get("confidence_score"),
        )

    human_content = _build_human_message(state, low_confidence)

    messages_to_llm = [
        SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = get_llm().invoke(messages_to_llm)  # CHANGED: lazy llm
    report = (
        str(response.content).strip()
        if hasattr(response, "content")
        else str(response).strip()
    )

    logger.info(  # CHANGED: was print()
        "Report generated (%d chars, low_confidence=%s)", len(report), low_confidence,
    )

    return {"messages": [AIMessage(content=report)]}
