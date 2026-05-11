"""
LangGraph workflow assembler for the business research assistant.

Full pipeline
-------------

    START
      |
      v
  clarity_agent  ─── needs_clarification ──>  human_feedback
      ^                                              |
      └──────────────── loops back ─────────────────┘
      |
    clear
      |
      v
  research_agent
      |
      v  (always validates regardless of confidence score)
  validator_agent
      |
      ├── insufficient + attempts < 3 ──> research_agent  (retry loop)
      |
      └── sufficient  OR  attempts >= 3 ──> synthesis_agent
                                                  |
                                                  v
                                                 END

Node registry
-------------
Node name         | Function         | Responsibility
------------------|------------------|-----------------------------------------
clarity_agent     | clarity_agent()  | Decide if query is specific enough
human_feedback    | human_feedback() | Pause for user clarification (interrupt)
research_agent    | research_agent() | Run Tavily searches, synthesise findings
validator_agent   | validator_agent()| Gate: are findings sufficient?
synthesis_agent   | synthesis_agent()| Write the final user-facing report

Interrupt / human-in-the-loop
------------------------------
The ``human_feedback`` node calls ``langgraph.types.interrupt()``.

On first execution the graph pauses and surfaces an interrupt payload to the
caller.  The caller (``main.py``) detects the ``__interrupt__`` event in the
stream, prompts the user on stdin, then resumes the graph by passing a
``Command(resume=<user_text>)`` with the same ``thread_id`` config.

On resumption LangGraph re-runs the node from the top; this time
``interrupt()`` returns the user's text instead of raising ``GraphInterrupt``.
The node appends a ``HumanMessage`` and resets ``clarity_status`` to ``None``
so the clarity agent re-evaluates the refined query.

Checkpointing
-------------
A ``SqliteSaver`` checkpointer writes every checkpoint to a local SQLite file
(``data/conversations.db`` relative to the project root).  Conversation history
therefore survives process restarts — the graph resumes exactly where it left
off for a given ``thread_id``.

# PRODUCTION: swap SqliteSaver for PostgresSaver(conn=async_pg_pool) here.

Note: ``GraphState`` is the correct state class name — ``ResearchState`` does
not exist in this codebase.
"""

import logging
import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from graph.nodes.clarity_agent import clarity_agent
from graph.nodes.research_agent import research_agent
from graph.nodes.synthesis_agent import synthesis_agent
from graph.nodes.validator_agent import validator_agent
from graph.state import GraphState

logger = logging.getLogger(__name__)

# Maximum number of research passes before forcing synthesis regardless of
# validation verdict.  Prevents an infinite research ↔ validator loop.
_MAX_RESEARCH_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Human-feedback node
# ---------------------------------------------------------------------------

def human_feedback_node(state: GraphState) -> dict:
    """Pause graph execution and request clarification from the user.

    Behaviour
    ---------
    *First execution:* ``interrupt()`` raises ``GraphInterrupt``, serialises
    the payload to the checkpointer, and halts the graph.  The caller receives
    an ``__interrupt__`` event in the stream.

    *After resumption:* the node re-runs from the top; ``interrupt()`` now
    returns whatever value the caller passed in ``Command(resume=...)``.
    That value is treated as the user's clarification text.

    State mutations returned
    ------------------------
    - ``messages``       — new ``HumanMessage`` with the clarification text
    - ``original_query`` — overwritten with the clarification so downstream
                           nodes (clarity, research) work with the refined ask
    - ``clarity_status`` — reset to ``None`` so ``clarity_agent`` re-evaluates
                           the new query from scratch
    """
    original = (state.get("original_query") or "").strip()

    # Pause here on first run; return user's text on resumption.
    user_input: str = interrupt(
        {
            "message": (
                "Could you clarify your query? "
                "Please specify the company name and what you'd like to know."
            ),
            "original_query": original,
        }
    )

    clarification = str(user_input).strip() if user_input else ""

    logger.info("Human clarification received: %r", clarification)

    # Do not overwrite original_query here: clarity_agent re-combines all recent
    # HumanMessages (including this clarification) into combined_query and updates
    # original_query downstream.
    update: dict = {
        "messages": [HumanMessage(content=clarification or "(no clarification provided)")],
        "clarity_status": None,  # Reset so clarity_agent re-evaluates from scratch.
    }
    return update

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_from_clarity(state: GraphState) -> str:
    """Route after clarity_agent based on whether the query is actionable.

    Returns
    -------
    "human_feedback"  — query is vague / missing entity; ask user to clarify
    "research_agent"  — query is specific enough; proceed to web research
    """
    if state.get("clarity_status") == "needs_clarification":
        # Query is too vague, generic, or missing a named entity.
        # Interrupt and ask the user for a more specific question.
        return "human_feedback"
    # clarity_status == "clear" (or unexpected value) — safe to research.
    return "research_agent"

def _route_from_research(state: GraphState) -> str:
    """Route after research_agent — always proceeds to validator.

    The conditional edge is kept explicit (rather than a simple add_edge) to
    document intent and make it trivial to add confidence-based branching
    (e.g. skip validation for very high-confidence results) in the future.
    """
    # Both branches lead to validation; confidence score is used by the
    # validator itself, not here, to keep routing concerns separated.
    return "validator_agent"

def _route_from_validator(state: GraphState) -> str:
    """Route after validator_agent — either retry research or synthesise.

    Retry condition (ALL must be true):
      - ``validation_result`` == "insufficient"
      - ``research_attempts`` < _MAX_RESEARCH_ATTEMPTS

    Otherwise go straight to synthesis (includes: sufficient verdict, OR max
    attempts reached — we generate the best report we can from what we have).

    Returns
    -------
    "research_agent"   — retry with another round of Tavily searches
    "synthesis_agent"  — produce the final report
    """
    attempts = state.get("research_attempts") or 0
    insufficient = state.get("validation_result") == "insufficient"

    if insufficient and attempts < _MAX_RESEARCH_ATTEMPTS:
        # Still within the retry budget and findings are not good enough.
        return "research_agent"

    # Sufficient, OR we've exhausted retries — write the best report possible.
    return "synthesis_agent"

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Assemble, compile, and return the LangGraph research workflow.

    Returns a ``CompiledStateGraph`` with an in-memory checkpointer attached.
    Callers must supply a ``config={"configurable": {"thread_id": <str>}}``
    when invoking the graph so the checkpointer can persist interrupt state.

    Returns
    -------
    langgraph.graph.state.CompiledStateGraph
    """
    workflow = StateGraph(GraphState)

    # ------------------------------------------------------------------ nodes
    workflow.add_node("clarity_agent", clarity_agent)
    workflow.add_node("human_feedback", human_feedback_node)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("validator_agent", validator_agent)
    workflow.add_node("synthesis_agent", synthesis_agent)

    # ------------------------------------------------------------------ edges
    # Entry point: always start with a clarity check.
    workflow.add_edge(START, "clarity_agent")

    # After clarity: branch on whether the query needs human clarification.
    workflow.add_conditional_edges(
        "clarity_agent",
        _route_from_clarity,
        {
            "human_feedback": "human_feedback",
            "research_agent": "research_agent",
        },
    )

    # After human_feedback: loop straight back to clarity for re-evaluation.
    # clarity_status was reset to None in human_feedback_node, so the agent
    # treats the clarification as a fresh query.
    workflow.add_edge("human_feedback", "clarity_agent")

    # After research: always validate (both branches go to the same target).
    workflow.add_conditional_edges(
        "research_agent",
        _route_from_research,
        {"validator_agent": "validator_agent"},
    )

    # After validation: retry research OR proceed to synthesis.
    workflow.add_conditional_edges(
        "validator_agent",
        _route_from_validator,
        {
            "research_agent": "research_agent",
            "synthesis_agent": "synthesis_agent",
        },
    )

    # Synthesis is always the terminal node.
    workflow.add_edge("synthesis_agent", END)

    # ----------------------------------------- compile with SQLite checkpointer
    # SqliteSaver persists checkpoints to disk so conversation history survives
    # restarts. Database file: project-root/data/conversations.db.
    #
    # SqliteSaver.from_conn_string() is a context manager, so for a server that
    # must keep the connection open for its entire lifetime we open the raw
    # sqlite3 connection ourselves and pass it in directly — same pattern that
    # from_conn_string() uses internally.
    #
    # PRODUCTION: swap SqliteSaver for PostgresSaver(conn=async_pg_pool) here.
    db_path = Path(__file__).parent.parent.parent / "data" / "conversations.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn=_conn)
    return workflow.compile(checkpointer=checkpointer)
