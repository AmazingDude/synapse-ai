"""
Validator Agent — third node in the business research pipeline.

Acts as a quality gate between research and synthesis.  Asks the LLM whether
the findings are relevant, complete, and substantial enough to be turned into
an executive report.  Returns a binary verdict.
"""

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState
from utils.llm import get_llm
from utils.parsing import extract_outermost_json

logger = logging.getLogger(__name__)

_SCORE_THRESHOLD_LOW = 4.0
_SCORE_THRESHOLD_HIGH = 7.0

_VALIDATION_SYSTEM_PROMPT = """\
You are a research quality validator. Your job is ONLY to check if the
provided research findings adequately answer the specific question asked.
Ignore any previous conversation context — focus exclusively on whether
the research answers THIS specific question.

Respond with ONLY a valid JSON object (no markdown fences, no extra text):
{"validation_result": "sufficient" | "insufficient", "gaps": "<one sentence - what is missing, or 'none'>"}
"""  # CHANGED: replaced verbose multi-criteria prompt; validator no longer sees full history

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
    json_str = extract_outermost_json(raw)
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
            logger.warning("JSON parse error in validator response: %s", exc)

    lower = raw.lower()
    has_insufficient = "insufficient" in lower
    has_sufficient = "sufficient" in lower

    if has_sufficient and not has_insufficient:
        logger.warning("Fallback keyword match -> sufficient")
        return "sufficient", "Inferred from keyword scan (JSON parse failed)."

    logger.warning("Fallback default -> insufficient")
    return "insufficient", "Could not parse model response; defaulting to insufficient."

def validator_agent(state: GraphState) -> dict:
    """Decide whether research_findings is sufficient for synthesis."""
    findings = (state.get("research_findings") or "").strip()
    confidence = state.get("confidence_score")

    if not findings:
        logger.warning("No research_findings in state -> insufficient")
        return {"validation_result": "insufficient"}

    # Extract the LATEST human message — not the combined multi-turn original_query.  # CHANGED
    # Using original_query caused topic-switch loops (e.g. Apple → NVIDIA) because  # CHANGED
    # it contained all prior human turns and the validator checked against them all.  # CHANGED
    latest_human_message = ""  # CHANGED
    for msg in reversed(state["messages"]):  # CHANGED
        if isinstance(msg, HumanMessage):  # CHANGED
            latest_human_message = str(msg.content).strip()  # CHANGED
            break  # CHANGED

    logger.info("Validating against latest question: %r", latest_human_message[:80])  # CHANGED

    human_content = (  # CHANGED: only current question + findings; no history context
        f"Question: {latest_human_message}\n\n"  # CHANGED
        f"Research findings:\n{findings}"  # CHANGED
    )  # CHANGED

    messages = [
        SystemMessage(content=_VALIDATION_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = get_llm().invoke(messages)
    raw_text = str(response.content) if hasattr(response, "content") else str(response)

    verdict, gaps = _parse_validation_response(raw_text)
    logger.info("verdict=%s confidence=%s gaps=%r", verdict, confidence, gaps)  # CHANGED

    return {"validation_result": verdict}
