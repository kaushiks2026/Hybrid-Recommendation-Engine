import pytest
import pandas as pd
from src.hybrid import HybridRecommender


def test_hybrid_alpha_calculation(fitted_models):
    hybrid = fitted_models["hybrid"]

    # Threshold is 5 in fixture
    # User 1 has 6 ratings -> alpha should be min(6/5, 1.0) = 1.0 (warm)
    assert hybrid._get_alpha(1) == 1.0

    # User 2 has 2 ratings -> alpha should be min(2/5, 1.0) = 0.4 (cold)
    assert hybrid._get_alpha(2) == 0.4

    # Unknown user (0 ratings) -> alpha should be 0.0 (pure CB)
    assert hybrid._get_alpha(999) == 0.0


def test_hybrid_score_normalization():
    raw_scores = {1: 1.0, 2: 3.0, 3: 5.0}
    norm = HybridRecommender._normalize_scores(raw_scores)
    assert norm[1] == 0.0
    assert norm[2] == 0.5
    assert norm[3] == 1.0

    # Constant scores case
    const_scores = {1: 4.0, 2: 4.0}
    norm_const = HybridRecommender._normalize_scores(const_scores)
    assert norm_const[1] == 0.5 and norm_const[2] == 0.5


def test_hybrid_get_top_k(fitted_models):
    hybrid = fitted_models["hybrid"]
    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    top_5 = hybrid.get_top_k(user_id=1, candidate_items=candidates, k=5)
    assert len(top_5) == 5
    # Scores in descending order
    for i in range(len(top_5) - 1):
        assert top_5[i][1] >= top_5[i + 1][1]


def test_hybrid_get_blend_stats(fitted_models, sample_ratings_df):
    hybrid = fitted_models["hybrid"]
    test_df = pd.DataFrame({"userId": [1, 2, 3]})
    stats_df = hybrid.get_blend_stats(test_df)

    assert "userId" in stats_df.columns
    assert "alpha" in stats_df.columns
    assert len(stats_df) == 3
