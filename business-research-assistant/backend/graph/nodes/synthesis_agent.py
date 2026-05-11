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

import logging
import os  # CHANGED: needed to read GROQ_API_KEY for synthesis-specific LLM
from functools import lru_cache  # CHANGED: singleton pattern for synthesis LLM

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq  # CHANGED: synthesis uses its own higher-token instance

from graph.state import GraphState  # CHANGED: get_llm removed; synthesis uses _get_synthesis_llm

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)  # CHANGED: dedicated synthesis LLM with higher token budget
def _get_synthesis_llm() -> ChatGroq:  # CHANGED
    """Groq instance used only by synthesis_agent.

    Uses max_tokens=4096 so the model can produce full 400-600 word reports
    without being cut off mid-section.  Other agents use the shared get_llm()
    which has no explicit token cap.
    """  # CHANGED
    return ChatGroq(  # CHANGED
        model="llama-3.3-70b-versatile",  # CHANGED
        temperature=0.3,  # CHANGED
        api_key=os.getenv("GROQ_API_KEY"),  # CHANGED: env already loaded by utils/llm.py
        max_tokens=4096,  # CHANGED: allows detailed multi-section reports
    )  # CHANGED


# Confidence < this triggers the visible low-confidence warning at the top.
_LOW_CONFIDENCE_THRESHOLD = 6.0

# How many recent messages to forward as conversation context.
_CONTEXT_MESSAGE_COUNT = 8

# Queries with more than this many HumanMessages are treated as follow-ups.
_FOLLOWUP_THRESHOLD = 2

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
- Write detailed, substantive paragraphs — minimum 3-4 sentences per section.
- Include specific numbers, dates, names, and figures from the research.
- Do NOT write one-sentence sections. Do NOT use vague language.
- Each section must contain concrete facts from the research data provided.
- Target length: 400-600 words for the full report.
- Only state facts supported by the research findings provided.
- Do not fabricate figures, dates, or events.
- If a section genuinely has no data, say so clearly rather than padding.
- Use the conversation history to handle follow-up questions correctly
  (e.g. when the user says "what about their competitors", "they" refers to
  the company discussed earlier in the conversation).
- Write in flowing prose for the first four sections; bullets only in
  Key Takeaways.
"""  # CHANGED: added detail/length rules to prevent thin one-sentence sections

_FOLLOWUP_SYSTEM_PROMPT = """\
This is a follow-up question in an ongoing research conversation.
Do NOT repeat the company overview — the user already received a full report
in a previous turn.  Focus the report ONLY on answering the specific
follow-up question based on the new research findings.

USE THE SAME FIVE SECTION HEADERS but populate only sections that contain
new information relevant to the follow-up question:

## 🏢 Company Overview     — OMIT unless the follow-up explicitly requests it
## 📰 Recent News & Developments
## 💰 Financial Snapshot
## 👥 Leadership & Strategy
## 🔑 Key Takeaways

If a section has nothing new to contribute for this specific follow-up,
write a single line under it:
> No new information for this section in the context of the follow-up question.

LOW-CONFIDENCE DISCLAIMER
-------------------------
If the inputs include "Low confidence flag: true", you MUST begin the report
(BEFORE the first ## header) with EXACTLY this blockquote line:

> ⚠️ Data confidence is low. Some information may be incomplete or outdated.

MANDATORY CLOSING NOTE
----------------------
Always end the report with this exact single line:

_Sources note: information above is drawn from public web sources and may not reflect real-time market data._

RULES
-----
- Write detailed, substantive paragraphs — minimum 3-4 sentences per section.
- Include specific numbers, dates, names, and figures from the research.
- Do NOT write one-sentence sections. Do NOT use vague language.
- Each section must contain concrete facts from the research data provided.
- Target length: 400-600 words for the full report.
- Focus tightly on what the follow-up question is actually asking.
- Use the conversation history to resolve references such as "they", "their",
  "the company" — they refer to the entity discussed earlier in the thread.
- Only state facts supported by the research findings provided.
- Do not fabricate figures, dates, or events.
- Write in flowing prose; bullets only in Key Takeaways.
"""  # CHANGED: added detail/length rules to follow-up prompt too

def _is_low_confidence(state: GraphState) -> bool:
    """Return True when the low-confidence disclaimer should be added."""
    score = state.get("confidence_score")
    if score is not None and score < _LOW_CONFIDENCE_THRESHOLD:
        return True
    if state.get("validation_result") == "insufficient":
        return True
    return False

def _is_followup(state: GraphState) -> bool:
    """Return True when this is a follow-up turn (more than 2 HumanMessages in history).

    A follow-up turn means the user has already received at least one full
    report, so the synthesis agent should avoid repeating the company overview
    and instead focus on the specific follow-up question.
    """
    msgs = state.get("messages") or []
    human_count = sum(1 for m in msgs if isinstance(m, HumanMessage))
    return human_count > _FOLLOWUP_THRESHOLD

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

def _build_human_message(
    state: GraphState,
    low_confidence: bool,
    is_followup: bool = False,
) -> str:
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
        f"Low confidence flag: {'true' if low_confidence else 'false'}",
        f"Follow-up turn: {'true' if is_followup else 'false'}",
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
    low_confidence = _is_low_confidence(state)
    is_followup = _is_followup(state)

    if low_confidence:
        logger.info(
            "Low-confidence signal (validation=%r, confidence=%s); disclaimer requested.",
            state.get("validation_result"),
            state.get("confidence_score"),
        )

    if is_followup:
        msgs = state.get("messages") or []
        human_count = sum(1 for m in msgs if isinstance(m, HumanMessage))
        logger.info(
            "Follow-up turn detected (%d human messages); using focused follow-up prompt.",
            human_count,
        )

    system_prompt = _FOLLOWUP_SYSTEM_PROMPT if is_followup else _SYNTHESIS_SYSTEM_PROMPT

    human_content = _build_human_message(state, low_confidence, is_followup)

    messages_to_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    response = _get_synthesis_llm().invoke(messages_to_llm)  # CHANGED: uses 4096-token instance
    report = (
        str(response.content).strip()
        if hasattr(response, "content")
        else str(response).strip()
    )

    logger.info(
        "Report generated (%d chars, low_confidence=%s)", len(report), low_confidence,
    )

    return {"messages": [AIMessage(content=report)]}
