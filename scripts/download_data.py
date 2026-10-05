"""
Download MovieLens 1M dataset.

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import download_movielens_1m

if __name__ == "__main__":
    download_movielens_1m(force=False)
    print("[OK] Data download complete!")
