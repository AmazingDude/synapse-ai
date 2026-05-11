"""
Groq LLM initialisation for the business research assistant.

Lazy-singleton pattern: the chat model is constructed on the first call to
``get_llm()`` and reused thereafter (cached by ``functools.lru_cache``).
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# This file lives under backend/utils/; two parent segments reach backend/,
# and one more reaches the project root where .env lives.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Return the shared Groq chat model, creating it on first call.

    Reads ``GROQ_API_KEY`` (required) from the environment.

    Raises
    ------
    RuntimeError
        If ``GROQ_API_KEY`` is not set in the environment / ``.env``.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    logger.info("Initialising Groq model: llama-3.3-70b-versatile")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=api_key,
    )
