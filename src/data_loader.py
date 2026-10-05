"""
Data loader for MovieLens 1M dataset.

Downloads the dataset from GroupLens if not already present and loads it
into pandas DataFrames.
"""

import io
import os
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import DATA_DIR, get_logger, ensure_dirs

logger = get_logger("data_loader")

# MovieLens 1M URL (GroupLens official)
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(force: bool = False) -> None:
    """Download and extract MovieLens 1M dataset."""
    ensure_dirs()
    target_dir = DATA_DIR / "ml-1m"

    if target_dir.exists() and not force:
        logger.info(f"Dataset already exists at {target_dir}")
        return

    zip_path = DATA_DIR / "ml-1m.zip"

    logger.info(f"Downloading MovieLens 1M from {ML1M_URL} ...")
    response = requests.get(ML1M_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info("Extracting archive ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    # Clean up the zip file
    os.remove(zip_path)
    logger.info(f"Dataset extracted to {target_dir}")


def load_ratings() -> pd.DataFrame:
    """
    Load ratings.dat → DataFrame with columns:
      userId, movieId, rating, timestamp
    """
    path = DATA_DIR / "ml-1m" / "ratings.dat"
    logger.info(f"Loading ratings from {path}")

    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(df):,} ratings from {df['userId'].nunique():,} users "
                f"on {df['movieId'].nunique():,} movies")
    return df


def load_movies() -> pd.DataFrame:
    """
    Load movies.dat → DataFrame with columns:
      movieId, title, genres
    """
    path = DATA_DIR / "ml-1m" / "movies.dat"
    logger.info(f"Loading movies from {path}")

    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(df):,} movies")
    return df


def load_users() -> pd.DataFrame:
    """
    Load users.dat → DataFrame with columns:
      userId, gender, age, occupation, zipCode
    """
    path = DATA_DIR / "ml-1m" / "users.dat"
    logger.info(f"Loading users from {path}")

    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["userId", "gender", "age", "occupation", "zipCode"],
        engine="python",
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(df):,} users")
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three DataFrames: (ratings, movies, users)."""
    download_movielens_1m()
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    return ratings, movies, users
