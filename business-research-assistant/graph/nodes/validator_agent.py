"""
Validator Agent — third node in the business research pipeline.

Acts as a quality gate between research and synthesis.  Asks the LLM whether
the findings are relevant, complete, and substantial enough to be turned into
an executive report.  Returns a binary verdict.
"""

import json
import logging  # CHANGED: replaced print() with logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState
from utils.llm import get_llm  # CHANGED: lazy singleton
from utils.parsing import extract_outermost_json  # CHANGED: shared parser

logger = logging.getLogger(__name__)  # CHANGED

_SCORE_THRESHOLD_LOW = 4.0
_SCORE_THRESHOLD_HIGH = 7.0

_VALIDATION_SYSTEM_PROMPT = """\
You are a senior research quality analyst for a business intelligence assistant.

Your task is to decide whether a set of research findings is good enough to
support writing an executive-level answer to the user's original query.

Evaluate the findings against three criteria:

1. RELEVANCE
   Do the findings directly address what the user asked?  Search results that
   drift to competitors, unrelated industries, or generic background without
   touching the user's specific question are NOT relevant enough.

2. COMPLETENESS
   Are there major factual gaps?  Examples of critical gaps:
   - No financial figures when the user asked about revenue or valuation.
   - No recent events when the query is about current/recent activity.
   - Only generic descriptions with no concrete facts, dates, or numbers.
   - Data that appears to be older than 2 years and is likely outdated.

3. SUBSTANCE
   Is there enough concrete detail (specific numbers, dates, named people or
   products, sourced events) to write at least two solid paragraphs that
   would satisfy a business analyst?

Decision rules:
  - If all three criteria pass -> "sufficient"
  - If any one criterion clearly fails -> "insufficient"
  - When in doubt, lean towards "insufficient" to trigger better research.

Respond with ONLY a valid JSON object (no markdown fences, no extra text):
{"validation_result": "sufficient" | "insufficient", "gaps": "<one sentence - what is missing, or 'none'>"}
"""


def _interpret_score(score: float | None) -> str:
    if score is None:
        return "No confidence score available; judge solely on content."
    if score < _SCORE_THRESHOLD_LOW:
        return (
            f"The research agent assigned a LOW confidence score of {score:.1f}/10. "
            "This strongly suggests data is sparse or off-topic.  Default to "
            "'insufficient' unless you find compelling evidence otherwise."
        )
    if score < _SCORE_THRESHOLD_HIGH:
        return (
            f"The research agent assigned a BORDERLINE confidence score of {score:.1f}/10. "
            "Examine the findings carefully against all three criteria."
        )
    return (
        f"The research agent assigned a HIGH confidence score of {score:.1f}/10. "
        "Verify there are no critical gaps before confirming 'sufficient'."
    )


def _parse_validation_response(
    raw: str,
) -> tuple[Literal["sufficient", "insufficient"], str]:
    """Three-stage parser: JSON -> keyword scan -> default insufficient."""
    json_str = extract_outermost_json(raw)  # CHANGED: shared helper
    if json_str:
        try:
            data = json.loads(json_str)
            vr = str(data.get("validation_result", "")).strip().lower()
            gaps = str(data.get("gaps", "not specified")).strip()
            if vr == "sufficient":
                return "sufficient", gaps
            if vr == "insufficient":
                return "insufficient", gaps
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("JSON parse error in validator response: %s", exc)  # CHANGED

    lower = raw.lower()
    has_insufficient = "insufficient" in lower
    has_sufficient = "sufficient" in lower

    if has_sufficient and not has_insufficient:
        logger.warning("Fallback keyword match -> sufficient")  # CHANGED
        return "sufficient", "Inferred from keyword scan (JSON parse failed)."

    logger.warning("Fallback default -> insufficient")  # CHANGED
    return "insufficient", "Could not parse model response; defaulting to insufficient."


def validator_agent(state: GraphState) -> dict:
    """Decide whether research_findings is sufficient for synthesis."""
    original_query = (state.get("original_query") or "").strip()
    findings = (state.get("research_findings") or "").strip()
    confidence = state.get("confidence_score")

    if not findings:
        logger.warning("No research_findings in state -> insufficient")  # CHANGED
        return {"validation_result": "insufficient"}

    score_hint = _interpret_score(confidence)

    human_content = (
        f"Original user query:\n{original_query}\n\n"
        f"Confidence score hint:\n{score_hint}\n\n"
        f"Research findings to evaluate:\n{findings}"
    )

    messages = [
        SystemMessage(content=_VALIDATION_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = get_llm().invoke(messages)  # CHANGED: lazy llm
    raw_text = str(response.content) if hasattr(response, "content") else str(response)

    verdict, gaps = _parse_validation_response(raw_text)
    logger.info(  # CHANGED
        "verdict=%s confidence_hint=%s gaps=%r", verdict, confidence, gaps,
    )

    return {"validation_result": verdict}
