"""
FastAPI backend for the business research assistant.

Endpoints
---------
POST /api/research  — start a new research query; streams progress as SSE
POST /api/clarify   — resume a graph paused by a clarification interrupt; same SSE
GET  /api/health    — service liveness check

SSE event format
----------------
Each event is a single JSON object on a line prefixed with "data: ", followed
by a double newline per the SSE spec:

  data: {"type": "agent_start",         "agent": "clarity_agent", "label": "Evaluating query clarity..."}
  data: {"type": "agent_start",         "agent": "research_agent", "label": "Researching company data..."}
  data: {"type": "agent_start",         "agent": "validator_agent","label": "Validating research quality..."}
  data: {"type": "agent_start",         "agent": "synthesis_agent","label": "Generating report..."}
  data: {"type": "clarification_needed","message": "Could you clarify your query?"}
  data: {"type": "report",              "content": "## Company Overview\\n..."}
  data: {"type": "error",               "message": "Rate limit reached. Please wait."}
  data: {"type": "done"}

Flow
----
1. Client POSTs to /api/research with { "message": "...", "thread_id": "uuid" }
2. Server streams SSE events until one of:
   a. "done" — research finished, "report" event already sent
   b. "clarification_needed" — graph paused; client prompts user, then calls /api/clarify
3. Client POSTs to /api/clarify with { "clarification": "...", "thread_id": "uuid" }
4. Server resumes the graph and continues streaming until "done"

Run the server
--------------
  cd backend
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env BEFORE importing graph modules.
# This file lives at backend/api.py, so parent.parent reaches the project root.
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel

from graph.graph_builder import build_graph
from utils.logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Business Research Assistant API",
    description="LangGraph multi-agent business research powered by Gemini + Tavily.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Single compiled graph shared across all requests.  MemorySaver is keyed by
# thread_id, so concurrent requests with different thread_ids are safely
# isolated inside the same process-level MemorySaver store.
_graph = build_graph()

# Friendly per-node labels — mirrors NODE_LABELS in main.py but without emojis
# (the frontend can add its own iconography).
_NODE_LABELS: dict[str, str] = {
    "clarity_agent":   "Evaluating query clarity...",
    "human_feedback":  "Waiting for clarification...",
    "research_agent":  "Researching company data...",
    "validator_agent": "Validating research quality...",
    "synthesis_agent": "Generating report...",
}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    """Body for POST /api/research."""

    message: str    # The user's research question
    thread_id: str  # Unique session ID; used by MemorySaver for checkpointing

class ClarifyRequest(BaseModel):
    """Body for POST /api/clarify."""

    clarification: str  # The user's clarification text
    thread_id: str      # Must match the thread_id from the interrupted /api/research call

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    """Encode one SSE frame: ``data: <json>\\n\\n`` per the SSE spec."""
    return f"data: {json.dumps(payload)}\n\n"

def _extract_last_report(config: dict) -> str:
    """Return the most recent non-empty AIMessage content from checkpointed state."""
    try:
        final_state = _graph.get_state(config)
        messages = final_state.values.get("messages", []) if final_state else []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and str(getattr(msg, "content", "")).strip():
                return str(msg.content).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read final state for report extraction: %s", exc)
    return ""

def _stream_graph(
    initial_input: dict | Command,
    thread_id: str,
) -> Iterator[str]:
    """Sync generator that drives the LangGraph run and yields SSE-formatted strings.

    Emits one ``agent_start`` frame per node that executes, a
    ``clarification_needed`` frame if the graph pauses for human input, an
    ``error`` frame on failures, and a ``report`` frame on successful
    completion.  Always ends with a ``done`` frame.

    This is a *synchronous* generator.  FastAPI's ``StreamingResponse`` runs
    sync generators in a thread pool, keeping the async event loop unblocked.

    Args:
        initial_input: Either a full state dict (new query via /api/research)
                       or a ``Command(resume=...)`` (resuming via /api/clarify).
        thread_id:     MemorySaver checkpointer key identifying the session.
    """
    config = {"configurable": {"thread_id": thread_id}}
    had_error = False
    was_interrupted = False

    try:
        # stream_mode="updates" produces {node_name: state_delta} dicts plus
        # {"__interrupt__": (Interrupt,)} when the graph pauses.
        for event in _graph.stream(initial_input, config, stream_mode="updates"):

            # ---- interrupt: graph paused, waiting for human clarification ----
            if "__interrupt__" in event:
                interrupt_obj = event["__interrupt__"][0]
                payload = interrupt_obj.value
                message = (
                    payload.get(
                        "message",
                        "Please clarify your query — specify a company name or topic.",
                    )
                    if isinstance(payload, dict)
                    else str(payload)
                )
                yield _sse({"type": "clarification_needed", "message": message})
                was_interrupted = True
                break  # Stop streaming; client must call /api/clarify to resume.

            # ---- normal node update: emit an agent_start progress event ----
            for node_name in event.keys():
                label = _NODE_LABELS.get(node_name, node_name)
                yield _sse({
                    "type": "agent_start",
                    "agent": node_name,
                    "label": label,
                })

    except GraphInterrupt as exc:
        # Defensive: some LangGraph versions raise GraphInterrupt instead of
        # surfacing it as an __interrupt__ event inside the stream.
        payload = exc.args[0] if exc.args else {}
        if isinstance(payload, (list, tuple)) and payload:
            payload = payload[0].value if hasattr(payload[0], "value") else payload[0]
        message = (
            payload.get("message", "Please clarify your query.")
            if isinstance(payload, dict)
            else str(payload)
        )
        yield _sse({"type": "clarification_needed", "message": message})
        was_interrupted = True

    except Exception as exc:  # noqa: BLE001
        error_str = str(exc)
        logger.error("Graph execution failed: %s", exc)
        had_error = True

        # Surface rate-limit errors with a clear actionable message.
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            yield _sse({
                "type": "error",
                "message": (
                    "Rate limit reached (too many requests to Gemini). "
                    "Wait a minute and try again, or check your quota at: "
                    "https://ai.dev/rate-limit"
                ),
            })
        else:
            yield _sse({"type": "error", "message": f"Research failed: {exc}"})

    # Emit the final report only on a clean, non-interrupted, non-error run.
    if not had_error and not was_interrupted:
        report = _extract_last_report(config)
        if report:
            yield _sse({"type": "report", "content": report})

    # Always close the SSE stream with a "done" event so the client knows
    # the response is finished (even if an error or interrupt occurred).
    yield _sse({"type": "done"})

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/research")
def research(request: ResearchRequest) -> StreamingResponse:
    """Start a new research query and stream progress + final report as SSE.

    Initialises a fresh pipeline state for the given thread_id, resets all
    per-turn fields (clarity_status, research_findings, etc.) so prior turns
    on the same thread don't bleed into the new query, and streams agent
    progress events back to the client.

    If a clarification interrupt fires, the client receives a
    ``clarification_needed`` event followed by ``done``.  The client must then
    POST to ``/api/clarify`` with the same thread_id to resume.
    """
    logger.info(
        "POST /api/research thread_id=%s msg=%r",
        request.thread_id,
        request.message,
    )

    # Full initial state for this query turn.  Resetting pipeline fields
    # prevents clarity_status / research_findings from the previous turn
    # leaking into the current one.  The messages channel accumulates across
    # turns (via add_messages reducer) to preserve multi-turn context.
    initial_state: dict = {
        "messages": [HumanMessage(content=request.message)],
        "original_query": request.message,
        "research_attempts": 0,
        "clarity_status": None,
        "research_findings": None,
        "confidence_score": None,
        "validation_result": None,
        "clarification_response": None,
    }

    return StreamingResponse(
        _stream_graph(initial_state, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/api/clarify")
def clarify(request: ClarifyRequest) -> StreamingResponse:
    """Resume a graph that was paused waiting for human clarification.

    The client must pass the same thread_id used in the original
    ``/api/research`` call so MemorySaver can restore the interrupted state.
    ``Command(resume=clarification_text)`` tells LangGraph to continue from
    the interrupt point, passing the user's text to the ``human_feedback``
    node.
    """
    logger.info(
        "POST /api/clarify thread_id=%s clarification=%r",
        request.thread_id,
        request.clarification,
    )

    # Command(resume=...) is the LangGraph mechanism for resuming after an
    # interrupt.  The value is handed to interrupt() inside human_feedback_node.
    resume_command = Command(resume=request.clarification)

    return StreamingResponse(
        _stream_graph(resume_command, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/api/health")
def health() -> dict:
    """Liveness check — returns the active Gemini model name for quick verification."""
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    return {"status": "ok", "model": model}

# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
