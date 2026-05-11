"""
Gemini LLM initialisation for the business research assistant.

Lazy-singleton pattern: the chat model is constructed on the first call to
``get_llm()`` and reused thereafter (cached by ``functools.lru_cache``).  This
means importing this module never touches the network and never validates the
API key — the first agent that actually needs the model triggers initialisation
and gets a clear, actionable error if the key is missing.
"""

import logging  # CHANGED: replace ad-hoc prints with logging
import os
from functools import lru_cache  # CHANGED: lazy singleton
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)  # CHANGED

# Load .env eagerly so that get_llm() (called later, possibly from another
# module) can still read GOOGLE_API_KEY without each caller re-loading dotenv.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@lru_cache(maxsize=1)  # CHANGED: singleton via lru_cache
def get_llm() -> ChatGoogleGenerativeAI:  # CHANGED: was module-level llm = ...
    """Return the shared Gemini chat model, creating it on first call.

    Reads ``GOOGLE_API_KEY`` (required) and ``GEMINI_MODEL`` (optional; default
    ``gemini-2.5-flash``). Gemini free-tier limits apply per model id.

    Raises
    ------
    RuntimeError
        If ``GOOGLE_API_KEY`` is not set in the environment / ``.env``.
    """
    api_key = os.getenv("GOOGLE_API_KEY")  # CHANGED
    if not api_key:  # CHANGED: explicit key validation
        raise RuntimeError(  # CHANGED
            "GOOGLE_API_KEY not found in environment. Check your .env file."  # CHANGED
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info("Initialising Gemini model: %s", model_name)

    return ChatGoogleGenerativeAI(  # CHANGED
        model=model_name,  # CHANGED
        google_api_key=api_key,  # CHANGED
        temperature=0.3,
    )
