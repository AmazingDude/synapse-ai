"""
Groq LLM initialisation for the business research assistant.  # CHANGED: switched from Gemini to Groq

Lazy-singleton pattern: the chat model is constructed on the first call to
``get_llm()`` and reused thereafter (cached by ``functools.lru_cache``).
"""

import logging  # CHANGED
import os  # CHANGED
from functools import lru_cache  # CHANGED
from pathlib import Path  # CHANGED

from dotenv import load_dotenv  # CHANGED
from langchain_groq import ChatGroq  # CHANGED: was ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)  # CHANGED

# This file lives under backend/utils/; two parent segments reach backend/,
# and one more reaches the project root where .env lives.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")  # CHANGED


@lru_cache(maxsize=1)  # CHANGED
def get_llm() -> ChatGroq:  # CHANGED: return type is now ChatGroq
    """Return the shared Groq chat model, creating it on first call.  # CHANGED

    Reads ``GROQ_API_KEY`` (required) from the environment.  # CHANGED

    Raises
    ------
    RuntimeError
        If ``GROQ_API_KEY`` is not set in the environment / ``.env``.
    """
    api_key = os.getenv("GROQ_API_KEY")  # CHANGED: was GOOGLE_API_KEY
    if not api_key:  # CHANGED
        raise RuntimeError("GROQ_API_KEY not found in .env")  # CHANGED

    logger.info("Initialising Groq model: llama-3.3-70b-versatile")  # CHANGED
    return ChatGroq(  # CHANGED: was ChatGoogleGenerativeAI
        model="llama-3.3-70b-versatile",  # CHANGED
        temperature=0.3,  # CHANGED
        api_key=api_key,  # CHANGED
    )
