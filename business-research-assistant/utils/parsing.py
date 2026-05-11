"""
Shared parsing helpers for the business research assistant.

This module centralises text-parsing utilities that multiple agents need so we
do not duplicate fragile regex / JSON-extraction logic across files.
"""
# CHANGED: new file — extracted from research_agent.py and validator_agent.py


def extract_outermost_json(text: str) -> str | None:
    """Return the outermost ``{...}`` block in *text*, or ``None`` if absent.

    Uses a depth counter rather than a regex so that nested braces inside the
    JSON value (e.g. markdown code fences, escaped strings, or embedded
    objects) do not confuse the parser.

    Example
    -------
    >>> extract_outermost_json('garbage {"a": {"b": 1}} trailing')
    '{"a": {"b": 1}}'
    >>> extract_outermost_json('no braces here') is None
    True
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
