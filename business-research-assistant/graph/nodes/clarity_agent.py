"""
Clarity Agent — first node in the business research pipeline.

Two bug fixes applied here
--------------------------
**BUG 1 — agent was too strict.**  Previous prompt rejected "Tell me about
Apple" as "too broad".  The system prompt is now strongly biased toward
"clear": if any named entity (company, brand, ticker, well-known product) is
present, the query is clear regardless of how broad the question itself is.

**BUG 2 — clarification loop ignored prior turns.**  Previously the agent
evaluated only the latest message.  Now it concatenates the *HumanMessage*
content of the last 4 messages into a single ``combined_query`` and evaluates
that, so "Tell me about Apple" followed by "what about that company" is
recognised as clearly about Apple.  ``original_query`` is set to the combined
text so downstream agents inherit the full intent.

Returned state channels
-----------------------
- ``clarity_status``  -- "clear" or "needs_clarification"
- ``original_query``  -- the combined query text (carries multi-turn context)
"""

import json
import logging  # CHANGED: replaced print() with logging
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage  # CHANGED: import BaseMessage

from graph.state import GraphState
from utils.llm import get_llm  # CHANGED: lazy llm singleton
from utils.parsing import extract_outermost_json  # CHANGED: shared parser

logger = logging.getLogger(__name__)  # CHANGED

# How many of the most-recent HumanMessages to include in the combined query.
# Scanning only HumanMessages (not total messages) means AI response turns
# — which can be very long and numerous — never push earlier user intent out
# of the window.
_MAX_HUMAN_TURNS = 6  # CHANGED: was _CONTEXT_WINDOW = 4 (total messages); now 6 human-only turns

# CHANGED: BUG 1 — fully rewritten system prompt with liberal "clear" rule and
#                  6-8 concrete few-shot examples covering both classes.
_SYSTEM_PROMPT = """\
You are a query-routing classifier for a business intelligence assistant.

Your ONLY job is to decide if the user's query identifies SOMETHING the
research team can search for, or if it is so vague that no search is possible.

A query is CLEAR if it mentions any of:
  - A specific company name ("Apple", "Stripe", "Goldman Sachs")
  - A well-known brand or product ("iPhone", "GPT-4", "AWS")
  - A ticker symbol ("AAPL", "TSLA", "MSFT")
  - A clearly identifiable named market/sector ("EV battery supply chain",
    "US regional banking sector")

Breadth of the question does NOT matter.  "Tell me about Apple",
"What's going on with Tesla?", "Apple", "Microsoft overview", or "How is
Stripe doing?" are ALL CLEAR — the research agent will figure out the angle.

A query NEEDS_CLARIFICATION only when NO identifiable entity is present.
Examples: "tell me about that company", "what about them", "the big tech
firm", "some startup in fintech", or unintelligible gibberish.

DEFAULT RULE: When in doubt, choose "clear". It is always better to attempt
research on a broad-but-named query than to frustrate the user with
unnecessary clarification.

MULTI-TURN RULE: If ANY message in the conversation history names a specific  # CHANGED: new rule
company or entity, treat the ENTIRE conversation as being about that company.  # CHANGED
A follow-up like "What about their competitors?" or "How are they doing?" is  # CHANGED
CLEAR if a company was named in an earlier turn — you are given the combined   # CHANGED
text of recent human turns, so look for a company name anywhere in that text.  # CHANGED

Few-shot examples
-----------------
Query: "Tell me about Apple"
-> {"status": "clear", "reason": "Apple is a specific company."}

Query: "AAPL Q1 2025 earnings"
-> {"status": "clear", "reason": "AAPL ticker + financial question."}

Query: "How is Stripe doing?"
-> {"status": "clear", "reason": "Stripe is a specific company."}

Query: "Give me an overview of Microsoft"
-> {"status": "clear", "reason": "Microsoft is a specific company."}

Query: "iPhone sales trend"
-> {"status": "clear", "reason": "iPhone is a well-known product."}

Conversation history: ["Tell me about Apple", "What about their competitors?"]  # CHANGED: new example
-> {"status": "clear", "reason": "Apple was named earlier in the conversation."}  # CHANGED

Query: "Tell me about that company"
-> {"status": "needs_clarification", "reason": "No company is named — 'that company' is undefined."}

Query: "What about them"
-> {"status": "needs_clarification", "reason": "'Them' has no referent in the conversation."}

Query: "Some big retailer doing well"
-> {"status": "needs_clarification", "reason": "Generic category with no named retailer."}

Output format
-------------
Respond with ONLY a single JSON object, no markdown:
{"status": "clear" | "needs_clarification", "reason": "<one short sentence>"}
"""


def _extract_combined_query(state: GraphState) -> str:  # CHANGED: BUG 2 — was _extract_latest_human_message
    """Return the last ``_MAX_HUMAN_TURNS`` HumanMessages joined as one string.

    Scans the ENTIRE message history but keeps only HumanMessage instances,
    so AI response turns (which can be very long and numerous) never push
    earlier user intent out of the window.  Taking the last 6 *human* turns
    instead of the last 4 *total* messages means "Tell me about Apple" is
    still present in the combined query even after several AI replies have
    been appended to the thread.

    Falls back to ``original_query`` if no human turns are found.
    """
    msgs: list[BaseMessage] = state.get("messages") or []  # CHANGED
    # CHANGED: collect ALL HumanMessages from the full history (skip AIMessages),
    #          then slice to the last _MAX_HUMAN_TURNS entries.
    all_human_texts = [  # CHANGED
        str(m.content).strip()  # CHANGED
        for m in msgs  # CHANGED: was msgs[-_CONTEXT_WINDOW:] — now the full list
        if isinstance(m, HumanMessage) and str(m.content).strip()  # CHANGED
    ]
    recent_human_texts = all_human_texts[-_MAX_HUMAN_TURNS:]  # CHANGED: take last 6 human turns
    if recent_human_texts:  # CHANGED
        return "\n".join(recent_human_texts)  # CHANGED
    return (state.get("original_query") or "").strip()  # CHANGED


def _parse_llm_response(
    raw: str,
) -> tuple[Literal["clear", "needs_clarification"], str]:
    """Extract status/reason from the LLM's JSON; default to clear on parse loss.

    Note: with the new prompt's "when in doubt choose clear" rule, the
    *fallback* on parse failure also leans clear — staying with the old
    "default to needs_clarification" would re-introduce BUG 1 from the parser
    side.  Genuine vagueness will still flow through because the LLM's actual
    JSON response will say so.
    """
    json_str = extract_outermost_json(raw)  # CHANGED: shared helper
    if not json_str:  # CHANGED
        logger.warning("Could not extract JSON from clarity response; defaulting to clear.")  # CHANGED
        return "clear", "Could not parse model response; defaulting to clear."  # CHANGED: was needs_clarification (BUG 1)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error in clarity response: %s", exc)  # CHANGED
        return "clear", "JSON decode error; defaulting to clear."  # CHANGED: was needs_clarification (BUG 1)

    status_raw = str(data.get("status", "")).strip().lower()
    reason = str(data.get("reason", "No reason provided.")).strip()

    if status_raw == "needs_clarification":  # CHANGED: only the explicit negative class
        return "needs_clarification", reason
    return "clear", reason  # CHANGED: everything else (including unknown values) -> clear


def clarity_agent(state: GraphState) -> dict:
    """Evaluate query clarity using the combined recent conversation context.

    Returns a partial state update with ``clarity_status`` and
    ``original_query`` (the combined text, so downstream agents see full intent).
    """
    combined_query = _extract_combined_query(state)  # CHANGED: BUG 2

    if not combined_query:  # CHANGED: empty input -> ask for clarification
        logger.warning("clarity_agent received no human input; routing to clarification.")  # CHANGED
        return {"clarity_status": "needs_clarification", "original_query": ""}  # CHANGED

    logger.info("Evaluating clarity for combined query: %r", combined_query)  # CHANGED

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Query to evaluate:\n{combined_query}"),  # CHANGED: combined
    ]

    response = get_llm().invoke(messages)  # CHANGED: lazy llm
    raw_text = str(response.content) if hasattr(response, "content") else str(response)

    status, reason = _parse_llm_response(raw_text)
    logger.info("Clarity verdict: status=%s, reason=%s", status, reason)  # CHANGED

    return {
        "clarity_status": status,
        "original_query": combined_query,  # CHANGED: BUG 2 — propagate combined query downstream
    }
