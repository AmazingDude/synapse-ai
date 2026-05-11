"""
Gemini LLM initialisation for the business research assistant.

Lazy-singleton pattern: the chat model is constructed on the first call to
``get_llm()`` and reused thereafter (cached by ``functools.lru_cache``).  This
means importing this module never touches the network and never validates the
API key — the first agent that actually needs the model triggers initialisation
and gets a clear, actionable error if the key is missing.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Load .env eagerly so that get_llm() (called later, possibly from another
# module) can still read GOOGLE_API_KEY without each caller re-loading dotenv.
# This file lives under backend/utils/, so three ``parent`` segments reach the project root.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return the shared Gemini chat model, creating it on first call.

    Reads ``GOOGLE_API_KEY`` (required) and ``GEMINI_MODEL`` (optional, defaults
    to ``gemini-2.0-flash-lite``) from the environment.

    Raises
    ------
    RuntimeError
        If ``GOOGLE_API_KEY`` is not set in the environment / ``.env``.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found in environment. Check your .env file."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    logger.info("Initialising Gemini model: %s", model_name)

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
    )
