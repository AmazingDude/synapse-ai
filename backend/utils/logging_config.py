"""
Centralised logging configuration for the business research assistant.

Call ``configure_logging()`` exactly once at process startup (see ``main.py``).
After that, every module can simply ``logger = logging.getLogger(__name__)``
and use ``.info``, ``.warning``, ``.error`` without further setup.

Conventions used across the codebase
------------------------------------
- ``logger.info``    — normal stage transitions and useful debug detail
- ``logger.warning`` — recoverable issues (e.g. JSON parse fell back)
- ``logger.error``   — API/network failures, missing keys, unhandled exceptions
"""

import logging
import sys

_CONFIGURED = False

def configure_logging(level: int = logging.INFO) -> None:
    """Initialise the root logger with a single console handler.

    Safe to call more than once; subsequent calls are no-ops so importing
    modules during tests can't accidentally reconfigure mid-run.

    Args:
        level: Default severity threshold (``logging.INFO`` by default).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers added by libraries on import so our format wins.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)

    _CONFIGURED = True
