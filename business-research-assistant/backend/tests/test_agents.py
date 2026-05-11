# Test suite for the research assistant backend agents and routing.
#
# Design principles:
#   - No real API calls: every LLM and search tool is mocked via unittest.mock.patch.
#   - Fast: all tests complete in well under a second (no network, no disk I/O).
#   - Isolated: each test patches only the specific module reference it exercises.
#   - Precise: assertions target exact return-value shapes, not just "truthy".

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from graph.nodes.clarity_agent import clarity_agent
from graph.nodes.research_agent import _extract_search_subject
from graph.graph_builder import (
    _route_from_clarity,
    _route_from_validator,
)
from utils.parsing import extract_outermost_json

# ===========================================================================
# Clarity Agent Tests
# ===========================================================================

class TestClarityAgent:
    """Tests for graph/nodes/clarity_agent.py."""

    def test_clarity_clear_with_company_name(self, mock_llm, mock_state):
        """A query containing a company name is classified as clear."""
        llm = mock_llm('{"status": "clear", "reason": "Apple is a specific company"}')
        with patch("graph.nodes.clarity_agent.get_llm", return_value=llm):
            result = clarity_agent(mock_state)

        assert result["clarity_status"] == "clear"
        assert result["original_query"] is not None

    def test_clarity_needs_clarification_no_company(self, mock_llm):
        """A query with no named entity is classified as needs_clarification."""
        state = {
            "messages": [HumanMessage(content="Tell me about that company")],
            "original_query": "Tell me about that company",
            "research_attempts": 0,
        }
        llm = mock_llm('{"status": "needs_clarification", "reason": "no company named"}')
        with patch("graph.nodes.clarity_agent.get_llm", return_value=llm):
            result = clarity_agent(state)

        assert result["clarity_status"] == "needs_clarification"

    def test_clarity_combines_human_messages(self, mock_llm, multi_turn_state):
        """The clarity agent combines ALL human turns — not just the latest one.

        With the multi_turn_state fixture, HumanMessages are:
          1. "Tell me about Apple"
          2. "Tell me about that company"
          3. "What about their competitors?"

        _extract_combined_query must include turn 1 so "Apple" appears in
        original_query, even though the last message alone contains no entity.
        """
        llm = mock_llm('{"status": "clear", "reason": "Apple named in history"}')
        with patch("graph.nodes.clarity_agent.get_llm", return_value=llm):
            result = clarity_agent(multi_turn_state)

        assert "Apple" in result["original_query"]

    def test_clarity_json_parse_failure_defaults_to_clear(self, mock_llm, mock_state):
        """When the LLM returns non-JSON, the fallback is 'clear', not 'needs_clarification'.

        This is intentional (BUG 1 fix): defaulting to 'clear' on a parse error
        prevents unnecessary clarification prompts — genuine vague queries still
        get routed correctly because the LLM returns valid JSON saying so.
        """
        llm = mock_llm("I cannot determine this")
        with patch("graph.nodes.clarity_agent.get_llm", return_value=llm):
            result = clarity_agent(mock_state)

        # NOTE: spec originally expected "needs_clarification" here, but the
        # BUG 1 fix changed the fallback to "clear" to avoid over-strict routing.
        assert result["clarity_status"] == "clear"

# ===========================================================================
# Routing Logic Tests
# ===========================================================================

class TestRoutingLogic:
    """Tests for the conditional-edge routing functions in graph/graph_builder.py."""

    def test_routing_clear_goes_to_research(self, mock_state):
        """clarity_status='clear' routes to research_agent."""
        state = {**mock_state, "clarity_status": "clear"}
        assert _route_from_clarity(state) == "research_agent"

    def test_routing_needs_clarification_goes_to_human_feedback(self, mock_state):
        """clarity_status='needs_clarification' routes to human_feedback."""
        state = {**mock_state, "clarity_status": "needs_clarification"}
        assert _route_from_clarity(state) == "human_feedback"

    def test_routing_validator_loops_if_insufficient_and_under_limit(self, mock_state):
        """Insufficient findings with attempts < MAX retry back to research_agent."""
        state = {
            **mock_state,
            "validation_result": "insufficient",
            "research_attempts": 1,
        }
        assert _route_from_validator(state) == "research_agent"

    def test_routing_validator_goes_to_synthesis_at_max_attempts(self, mock_state):
        """Insufficient findings at MAX attempts force synthesis (best-effort report)."""
        state = {
            **mock_state,
            "validation_result": "insufficient",
            "research_attempts": 3,
        }
        assert _route_from_validator(state) == "synthesis_agent"

    def test_routing_validator_goes_to_synthesis_if_sufficient(self, mock_state):
        """Sufficient findings always route to synthesis_agent regardless of attempt count."""
        state = {
            **mock_state,
            "validation_result": "sufficient",
            "research_attempts": 1,
        }
        assert _route_from_validator(state) == "synthesis_agent"

# ===========================================================================
# Parsing Utility Tests
# ===========================================================================

class TestParsingUtility:
    """Tests for utils/parsing.py — the shared JSON extraction helper."""

    def test_extract_json_valid_input(self):
        """A standalone JSON object is extracted and parseable."""
        raw = '{"key": "value"}'
        result = extract_outermost_json(raw)
        assert result is not None
        assert json.loads(result) == {"key": "value"}

    def test_extract_json_with_surrounding_text(self):
        """JSON embedded in surrounding prose is extracted correctly."""
        raw = 'Here is the result: {"status": "clear"} done'
        result = extract_outermost_json(raw)
        assert result is not None
        assert json.loads(result) == {"status": "clear"}

    def test_extract_json_nested_objects(self):
        """Nested braces inside the JSON value do not confuse the depth counter."""
        raw = 'prefix {"outer": {"inner": 1}} suffix'
        result = extract_outermost_json(raw)
        assert json.loads(result) == {"outer": {"inner": 1}}

    def test_extract_json_invalid_returns_none(self):
        """Input with no JSON object at all returns None."""
        result = extract_outermost_json("no json here")
        assert result is None

    def test_extract_json_empty_string_returns_none(self):
        """Empty string returns None without raising."""
        result = extract_outermost_json("")
        assert result is None

# ===========================================================================
# Research Agent Tests
# ===========================================================================

class TestResearchAgent:
    """Tests for graph/nodes/research_agent.py."""

    def test_search_subject_extraction_falls_back_on_llm_error(self):
        """When the LLM raises, _extract_search_subject falls back to the first line.

        Fallback logic: ``combined_query.split('\\n')[0][:50].strip()``
        So "Tell me about Apple\\nWhat about competitors?" → "Tell me about Apple".
        """
        with patch(
            "graph.nodes.research_agent.get_llm",
            side_effect=RuntimeError("GOOGLE_API_KEY not found"),
        ):
            result = _extract_search_subject(
                "Tell me about Apple\nWhat about competitors?"
            )

        assert result == "Tell me about Apple"

    def test_search_subject_extraction_returns_stripped_llm_response(self):
        """A valid LLM response is cleaned (whitespace/punctuation stripped) and returned."""
        response_mock = MagicMock()
        response_mock.content = "  Apple competitors!  "
        llm_mock = MagicMock()
        llm_mock.invoke.return_value = response_mock

        with patch("graph.nodes.research_agent.get_llm", return_value=llm_mock):
            result = _extract_search_subject("Tell me about Apple\nWhat about competitors?")

        assert result == "Apple competitors"

    def test_search_subject_fallback_truncates_long_first_line(self):
        """Fallback truncates first line to 50 characters when LLM fails."""
        long_query = "A" * 80 + "\nsecond line"
        with patch(
            "graph.nodes.research_agent.get_llm",
            side_effect=RuntimeError("api error"),
        ):
            result = _extract_search_subject(long_query)

        assert len(result) <= 50
        assert result == "A" * 50
