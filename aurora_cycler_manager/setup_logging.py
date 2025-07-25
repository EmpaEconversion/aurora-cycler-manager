"""Copyright © 2025, Empa

Set up logging.
"""

import logging
import sys


def setup_logging(level=logging.INFO) -> None:
    """Set up logging config."""
    for handler in logging.root.handlers[:]:  # Remove existing handlers
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
