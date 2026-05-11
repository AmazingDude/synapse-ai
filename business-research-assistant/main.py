"""
Business Research Assistant — interactive CLI runner.

User-facing prints (banner, prompts, final report, divider lines) stay as
plain ``print()`` calls.  Everything else — agent stage details, parse
warnings, search failures — goes through the standard ``logging`` module
configured via ``utils.logging_config.configure_logging()``.

Run with:
    python main.py

Type "exit", "quit", "bye", or "q" to leave the session.
"""

from __future__ import annotations

import logging  # CHANGED: use logging instead of print for backend traces
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from utils.logging_config import configure_logging  # CHANGED: central logger setup

logger = logging.getLogger(__name__)  # CHANGED: module-level logger for _run_query error reporting


# CHANGED: ensure emoji / unicode in NODE_LABELS work on Windows cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # CHANGED
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # CHANGED


# CHANGED: friendly per-node progress labels (emoji + clear action verb).
NODE_LABELS: dict[str, str] = {
    "clarity_agent":   "🔍  Evaluating query clarity...",
    "human_feedback":  "💬  Waiting for clarification...",
    "research_agent":  "📊  Researching company data...",
    "validator_agent": "✅  Validating research quality...",
    "synthesis_agent": "📝  Generating report...",
}


def _print_banner() -> None:
    """Welcome banner shown once at session start."""
    print()
    print("=" * 60)
    print("   Business Research Assistant")
    print("   Powered by Gemini + Tavily + LangGraph")
    print("=" * 60)
    print("  Ask any business research question.")
    print('  Type "exit", "quit", or "bye" to end the session.')
    print("=" * 60)
    print()


def _print_node_label(node_name: str) -> None:  # CHANGED: simpler than diff-based stage detection
    """Print the user-friendly emoji label for *node_name*, if recognised."""
    label = NODE_LABELS.get(node_name)
    if label:
        print(label)


def _extract_final_report(app: Any, config: dict) -> str:
    """Return the most recent non-empty AIMessage from the checkpointed state."""
    final_state = app.get_state(config)
    messages: list = (
        final_state.values.get("messages", []) if final_state else []
    )
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and str(getattr(msg, "content", "")).strip():
            return str(msg.content).strip()
    return ""


# ---------------------------------------------------------------------------
# Per-query execution
# ---------------------------------------------------------------------------

def _run_query(app: Any, user_input: str, config: dict) -> None:  # CHANGED: rewritten for updates mode
    """Stream one research query through the graph and print the final report.

    Uses ``stream_mode="updates"`` so each event is keyed by the node name
    that just produced it — perfect for printing user-friendly progress
    labels without diffing state snapshots.
    """
    print(f"\n  Researching: {user_input!r}\n")

    # Build the state update for this turn.  Per-turn pipeline fields are
    # reset so verdicts/findings from a previous query don't leak forward,
    # while the ``messages`` channel keeps the full multi-turn history via
    # the ``add_messages`` reducer.
    query_state: dict = {
        "messages": [HumanMessage(content=user_input)],
        "original_query": user_input,
        "research_attempts": 0,
        "clarity_status": None,
        "research_findings": None,
        "confidence_score": None,
        "validation_result": None,
        "clarification_response": None,
    }

    current_input: dict | Command = query_state

    while True:
        interrupted = False

        try:
            # CHANGED: stream_mode="updates" -> events are {node_name: state_delta}
            #          or {"__interrupt__": (Interrupt,)}.
            for event in app.stream(current_input, config, stream_mode="updates"):

                # ---- interrupt: graph paused for human clarification ----
                if "__interrupt__" in event:
                    interrupt_obj = event["__interrupt__"][0]
                    payload = interrupt_obj.value
                    if isinstance(payload, dict):
                        prompt_msg = payload.get(
                            "message",
                            "Please clarify your query (add a company name or more detail):",
                        )
                    else:
                        prompt_msg = str(payload)

                    print("\n  [Clarification needed]")
                    print(f"  {prompt_msg}\n")
                    print("  Your clarification: ", end="", flush=True)
                    clarification = sys.stdin.readline().strip()
                    print()

                    current_input = Command(resume=clarification)
                    interrupted = True
                    break  # restart the streaming loop with Command(resume=...)

                # ---- normal node output: print friendly progress label ----
                for node_name in event.keys():  # CHANGED: direct node name from updates mode
                    _print_node_label(node_name)
                    # CHANGED: divider line just before the final report.
                    if node_name == "synthesis_agent":
                        print()
                        print("-" * 60)

        except GraphInterrupt as exc:
            # Defensive fallback: GraphInterrupt may bubble up if anything in
            # the streaming layer mishandles the event.  Treat it the same way.
            payload = exc.args[0] if exc.args else {}
            if isinstance(payload, (list, tuple)) and payload:
                payload = payload[0].value if hasattr(payload[0], "value") else payload[0]
            prompt_msg = (
                payload.get("message", "Please clarify your query:")
                if isinstance(payload, dict)
                else str(payload)
            )
            print(f"\n  [Clarification needed]\n  {prompt_msg}")
            print("  Your clarification: ", end="", flush=True)
            clarification = sys.stdin.readline().strip()
            current_input = Command(resume=clarification)
            interrupted = True

        except Exception as exc:  # noqa: BLE001
            error_str = str(exc)  # CHANGED: inspect error text to detect rate limits
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:  # CHANGED: rate-limit detection
                print("\n  ⚠️  Rate limit reached (too many requests to Gemini).")  # CHANGED
                print("  Wait a minute and try again, or check your quota at:")  # CHANGED
                print("  https://ai.dev/rate-limit\n")  # CHANGED
            else:  # CHANGED
                print(f"\n  ❌  Something went wrong: {exc}\n")  # CHANGED: single clean line, no stale report
            logger.error("Graph execution failed: %s", exc)  # CHANGED: was logging.getLogger(__name__).error inline
            return  # CHANGED: exit _run_query entirely — skips _extract_final_report so stale report is never printed

        if not interrupted:
            break

    # ---- extract and print the finished report ----
    report = _extract_final_report(app, config)
    if report:
        print(report)
        print("-" * 60)
    else:
        print("\n  [No report was generated. Check the logs above for errors.]")


# ---------------------------------------------------------------------------
# Session entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # CHANGED: configure logging FIRST so any module-level loggers from
    #          downstream imports use our format from the start.
    configure_logging(level=logging.INFO)

    load_dotenv(Path(__file__).resolve().parent / ".env")

    # Import after dotenv so .env is loaded before any module-level reads.
    from graph.graph_builder import build_graph

    app = build_graph()

    session_thread_id = str(uuid.uuid4())
    config: dict = {"configurable": {"thread_id": session_thread_id}}

    _print_banner()

    while True:
        try:
            print("You: ", end="", flush=True)
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            print("\n  Goodbye! Thanks for using the Business Research Assistant.")
            break

        _run_query(app, user_input, config)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
