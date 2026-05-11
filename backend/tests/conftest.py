# Run tests with:       pytest backend/tests/ -v
# Run with coverage:    pytest backend/tests/ -v --cov=backend
# All tests mock external APIs — no real API keys needed.

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Path setup — add backend/ to sys.path so imports like
# `from graph.nodes.clarity_agent import clarity_agent` resolve correctly
# when pytest is invoked from the project root:
#   pytest backend/tests/ -v
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Factory fixture — call it with a JSON string to get a mock LLM.

    Usage::

        def test_something(mock_llm, mock_state):
            llm = mock_llm('{"status": "clear", "reason": "Apple named"}')
            with patch("graph.nodes.clarity_agent.get_llm", return_value=llm):
                result = clarity_agent(mock_state)

    The returned mock satisfies:
      - ``mock.invoke(messages)`` returns a ``MagicMock`` whose ``.content``
        attribute equals the *content* string passed to the factory.
      - This mirrors the interface ``ChatGoogleGenerativeAI.invoke`` presents
        to the agent functions.

    Args:
        content: The string the mock LLM ``invoke()`` should return as
                 ``.content``.  Defaults to a valid "clear" JSON response.
    """
    def _factory(
        content: str = '{"status": "clear", "reason": "test default"}',
    ) -> MagicMock:
        response_mock = MagicMock()
        response_mock.content = content
        llm_mock = MagicMock()
        llm_mock.invoke.return_value = response_mock
        return llm_mock

    return _factory

@pytest.fixture
def mock_state() -> dict:
    """Base GraphState dict with sensible single-turn defaults.

    Suitable for tests that exercise a single user query.  The ``messages``
    list contains exactly one ``HumanMessage`` so ``_extract_combined_query``
    in the clarity agent produces ``"Tell me about Apple"`` as the combined
    query.
    """
    return {
        "messages": [HumanMessage(content="Tell me about Apple")],
        "clarity_status": None,
        "research_findings": None,
        "confidence_score": None,
        "validation_result": None,
        "research_attempts": 0,
        "original_query": "Tell me about Apple",
        "clarification_response": None,
        "search_subject": None,
    }

@pytest.fixture
def multi_turn_state() -> dict:
    """GraphState dict simulating a real 3-turn conversation.

    Designed to exercise ``_extract_combined_query``'s ability to scan ALL
    HumanMessages and combine them — verifying that "Apple" (mentioned in
    turn 1) is still present in the combined query even after two AI response
    turns have been appended.

    Message sequence::

        HumanMessage: "Tell me about Apple"
        AIMessage:    "<full Apple report>"
        HumanMessage: "Tell me about that company"
        AIMessage:    "<follow-up report>"
        HumanMessage: "What about their competitors?"
    """
    return {
        "messages": [
            HumanMessage(content="Tell me about Apple"),
            AIMessage(content="Apple Inc. is a technology company..."),
            HumanMessage(content="Tell me about that company"),
            AIMessage(content="Apple continues to lead..."),
            HumanMessage(content="What about their competitors?"),
        ],
        "clarity_status": None,
        "research_findings": None,
        "confidence_score": None,
        "validation_result": None,
        "research_attempts": 0,
        "original_query": "Tell me about Apple",
        "clarification_response": None,
        "search_subject": None,
    }
