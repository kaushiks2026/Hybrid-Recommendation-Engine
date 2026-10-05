import os
from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_ratings, load_movies, load_users
from src.utils import DATA_DIR


def test_load_ratings_structure():
    ratings_file = DATA_DIR / "ml-1m" / "ratings.dat"
    if not ratings_file.exists():
        pytest.skip("ratings.dat not found in data/ml-1m")
    
    df = load_ratings()
    assert isinstance(df, pd.DataFrame)
    assert set(["userId", "movieId", "rating", "timestamp"]).issubset(df.columns)
    assert len(df) > 0
    assert (df["rating"] >= 1.0).all() and (df["rating"] <= 5.0).all()


def test_load_movies_structure():
    movies_file = DATA_DIR / "ml-1m" / "movies.dat"
    if not movies_file.exists():
        pytest.skip("movies.dat not found in data/ml-1m")

    df = load_movies()
    assert isinstance(df, pd.DataFrame)
    assert set(["movieId", "title", "genres"]).issubset(df.columns)
    assert len(df) > 0


def test_load_users_structure():
    users_file = DATA_DIR / "ml-1m" / "users.dat"
    if not users_file.exists():
        pytest.skip("users.dat not found in data/ml-1m")

    df = load_users()
    assert isinstance(df, pd.DataFrame)
    assert set(["userId", "gender", "age", "occupation", "zipCode"]).issubset(df.columns)
    assert len(df) > 0
