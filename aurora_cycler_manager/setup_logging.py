"""Copyright © 2025, Empa.

Set up logging.
"""

import logging
import sys


def setup_logging(level: int = logging.WARNING, aurora_level: int = logging.INFO) -> None:
    """Set up logging config."""
    for handler in logging.root.handlers[:]:  # Remove existing handlers
        logging.root.removeHandler(handler)

    # Aurora should be at least the general level
    aurora_level = min(aurora_level, level)

    # For all other packages
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # For aurora logging
    logging.getLogger("aurora_cycler_manager").setLevel(aurora_level)
