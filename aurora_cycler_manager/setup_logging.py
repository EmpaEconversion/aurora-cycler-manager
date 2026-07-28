# Copyright © 2025, Empa.
"""Set up logging."""

import logging
import sys


class _NoMillisecondsFormatter(logging.Formatter):
    """Formatter that removes milliseconds from the timestamp."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        """Return the creation time of the specified LogRecord as formatted text."""
        return super().formatTime(record, datefmt=datefmt or "%Y-%m-%d %H:%M:%S")


def setup_logging(level: int = logging.WARNING, aurora_level: int = logging.INFO) -> None:
    """Set up logging config."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    aurora_level = min(aurora_level, level)

    # Root handler: full format, no milliseconds
    root_handler = logging.StreamHandler(sys.stdout)
    root_handler.setFormatter(_NoMillisecondsFormatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.root.setLevel(level)
    logging.root.addHandler(root_handler)

    # Aurora handler: no %(name)s, propagation disabled so root handler is skipped
    # Clear old handlers
    aurora_logger = logging.getLogger("aurora_cycler_manager")
    for handler in aurora_logger.handlers[:]:
        aurora_logger.removeHandler(handler)

    aurora_handler = logging.StreamHandler(sys.stdout)
    aurora_handler.setFormatter(_NoMillisecondsFormatter(fmt="%(asctime)s [%(levelname)s] %(message)s"))
    aurora_logger = logging.getLogger("aurora_cycler_manager")
    aurora_logger.setLevel(aurora_level)
    aurora_logger.addHandler(aurora_handler)
    aurora_logger.propagate = False
