"""Mərkəzləşdirilmiş logging konfiqurasiyası."""

import logging
import sys

from .config import CONFIG


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("web_agent")
    if logger.handlers:
        return logger  # artıq qurulub

    logger.setLevel(CONFIG.log_level)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = setup_logging()
