"""
Shared LangGraph state for the multi-agent business research assistant.

Defines every channel producers and consumers coordinate on: conversational history
(messages, merged LangGraph‑style via ``add_messages``), clarity routing, synthesized
research, validation verdicts with confidence, retry counting, plus the originating
question and clarification text. Nodes should return partial updates; reducers combine
them between steps.
"""

from typing import Annotated, Literal, NotRequired, Required, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict, total=False):
    # Incoming turns (HumanMessage), agent replies (AIMessage), and optional role/system
    # messages accumulate here. Uses LangGraph’s ``add_messages`` reducer (same contract as
    # ``MessagesState``); do **not** replace it with naive ``operator.add`` on bare lists—
    # ``add_messages`` merges/replaces fragments correctly across turns.
    messages: Required[Annotated[list[BaseMessage], add_messages]]

    # Whether the user’s objective is actionable as-is or clarifying questions must run first.
    clarity_status: Literal["clear", "needs_clarification"] | None

    # Consolidated Tavily-derived summary (and intermediates distilled by the researcher).
    research_findings: str | None

    # Validator’s 0–10 confidence that findings cover the clarified brief adequately.
    confidence_score: float | None

    # High-level verdict on whether research depth/quality suffices for downstream synthesis.
    validation_result: Literal["sufficient", "insufficient"] | None

    # Count of finished research loops; callers should omit or set ``0``, then increments each pass.
    research_attempts: NotRequired[int]

    # Canonical user question at workflow start—persists across messages for bookkeeping.
    original_query: str | None

    # Clarified objectives, clarification Q&A transcript segment, or parser-extracted brief.
    clarification_response: str | None

    # LLM-extracted compact search subject, e.g. "Apple competitors" or "Tesla financials".
    # Set by research_agent; used for logging and potential future routing decisions.
    search_subject: str | None
