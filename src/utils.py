"""
Utility helpers: logging, configuration, and common functions.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ──────────────────────────────────────────────────────────────
# Configuration defaults
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "dataset": "ml-1m",                # "ml-1m" or "ml-25m"
    "cold_start_threshold": 5,         # users/items with < this many interactions are "cold"

    # Collaborative filtering (SVD)
    "svd_n_factors": 100,
    "svd_n_epochs": 20,
    "svd_lr_all": 0.005,
    "svd_reg_all": 0.02,

    # Hybrid blending
    "blend_threshold": 20,             # number of ratings at which CF gets full weight

    # Evaluation
    "top_k_values": [5, 10, 20],
    "relevance_threshold": 4.0,        # rating >= this is considered "relevant"

    # Diversity re-ranker (bonus)
    "mmr_lambda": 0.7,                 # balance between relevance and diversity

    # Reproducibility
    "random_seed": 42,
}


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ──────────────────────────────────────────────────────────────
# Timer context manager
# ──────────────────────────────────────────────────────────────
class Timer:
    """Simple context-manager timer for benchmarking steps."""

    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger or get_logger("timer")

    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.info(f">> Starting: {self.label}")
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.logger.info(f"[OK] Finished: {self.label} ({elapsed:.1f}s)")


def ensure_dirs():
    """Create project directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
