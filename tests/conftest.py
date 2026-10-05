import pytest
import pandas as pd
import numpy as np

from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter
from src.hybrid import HybridRecommender
from src.reranker import DiversityReranker


@pytest.fixture
def sample_movies_df():
    """Returns a sample movies DataFrame matching MovieLens 1M format."""
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Father of the Bride Part II (1995)",
            "Heat (1995)",
            "Sabrina (1995)",
            "Tom and Huck (1995)",
            "Sudden Death (1995)",
            "GoldenEye (1995)",
        ],
        "genres": [
            "Animation|Children's|Comedy",
            "Adventure|Children's|Fantasy",
            "Comedy|Romance",
            "Comedy|Drama",
            "Comedy",
            "Action|Crime|Thriller",
            "Comedy|Romance",
            "Adventure|Children's",
            "Action",
            "Action|Adventure|Thriller",
        ],
    })


@pytest.fixture
def sample_ratings_df():
    """
    Returns a sample ratings DataFrame with multiple users and timestamps.
    User 1: 6 ratings (warm)
    User 2: 2 ratings (cold)
    User 3: 4 ratings (cold)
    """
    data = [
        # User 1 (Warm user)
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 100},
        {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 101},
        {"userId": 1, "movieId": 3, "rating": 3.0, "timestamp": 102},
        {"userId": 1, "movieId": 4, "rating": 4.0, "timestamp": 103},
        {"userId": 1, "movieId": 5, "rating": 5.0, "timestamp": 104},
        {"userId": 1, "movieId": 6, "rating": 2.0, "timestamp": 105},
        # User 2 (Cold user)
        {"userId": 2, "movieId": 6, "rating": 5.0, "timestamp": 200},
        {"userId": 2, "movieId": 9, "rating": 4.0, "timestamp": 201},
        # User 3 (Cold user)
        {"userId": 3, "movieId": 1, "rating": 4.0, "timestamp": 300},
        {"userId": 3, "movieId": 5, "rating": 4.0, "timestamp": 301},
        {"userId": 3, "movieId": 7, "rating": 5.0, "timestamp": 302},
        {"userId": 3, "movieId": 8, "rating": 3.0, "timestamp": 303},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_users_df():
    """Returns a sample users DataFrame matching MovieLens 1M format."""
    return pd.DataFrame({
        "userId": [1, 2, 3],
        "gender": ["F", "M", "M"],
        "age": [1, 56, 25],
        "occupation": [10, 16, 15],
        "zipCode": ["48067", "70072", "55117"],
    })


@pytest.fixture
def fitted_models(sample_movies_df, sample_ratings_df):
    """Returns fitted CF, CB, Hybrid, and Reranker models for testing."""
    # Fast SVD model with few factors/epochs for unit testing
    cf = CollaborativeFilter(n_factors=5, n_epochs=5, random_state=42)
    cf.fit(sample_ratings_df)

    cb = ContentBasedFilter()
    cb.fit(sample_movies_df, sample_ratings_df)

    hybrid = HybridRecommender(cf, cb, blend_threshold=5)
    user_counts = sample_ratings_df.groupby("userId").size().to_dict()
    hybrid.set_user_counts(user_counts)

    reranker = DiversityReranker(cb, lam=0.7)

    return {
        "cf": cf,
        "cb": cb,
        "hybrid": hybrid,
        "reranker": reranker,
        "user_counts": user_counts,
    }
