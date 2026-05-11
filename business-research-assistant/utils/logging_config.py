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
# CHANGED: new file

import logging
import sys

_CONFIGURED = False  # CHANGED: idempotency guard


def configure_logging(level: int = logging.INFO) -> None:
    """Initialise the root logger with a single console handler.

    Safe to call more than once; subsequent calls are no-ops so importing
    modules during tests can't accidentally reconfigure mid-run.

    Args:
        level: Default severity threshold (``logging.INFO`` by default).
    """
    global _CONFIGURED  # CHANGED
    if _CONFIGURED:  # CHANGED
        return  # CHANGED

    root = logging.getLogger()  # CHANGED
    root.setLevel(level)  # CHANGED

    # Remove any handlers added by libraries on import so our format wins.
    for handler in list(root.handlers):  # CHANGED
        root.removeHandler(handler)  # CHANGED

    handler = logging.StreamHandler(stream=sys.stderr)  # CHANGED
    handler.setLevel(level)  # CHANGED
    handler.setFormatter(  # CHANGED
        logging.Formatter("%(levelname)s | %(name)s | %(message)s")  # CHANGED
    )
    root.addHandler(handler)  # CHANGED

    _CONFIGURED = True  # CHANGED
