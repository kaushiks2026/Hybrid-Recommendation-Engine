import pytest
from src.collaborative import CollaborativeFilter


def test_collaborative_filter_fit_and_predict(sample_ratings_df):
    cf = CollaborativeFilter(n_factors=5, n_epochs=5, random_state=42)
    cf.fit(sample_ratings_df)

    assert cf._trained is True
    assert cf.global_mean > 0

    score = cf.predict_score(user_id=1, item_id=1)
    assert isinstance(score, float)
    assert 1.0 <= score <= 5.0


def test_collaborative_filter_score_items(sample_ratings_df):
    cf = CollaborativeFilter(n_factors=5, n_epochs=5, random_state=42)
    cf.fit(sample_ratings_df)

    candidate_items = [1, 2, 3, 999]  # 999 is unknown item
    scores = cf.score_items(user_id=1, candidate_items=candidate_items)

    assert len(scores) == 4
    for item_id, s in scores.items():
        assert 1.0 <= s <= 5.0


def test_collaborative_filter_get_top_k(sample_ratings_df):
    cf = CollaborativeFilter(n_factors=5, n_epochs=5, random_state=42)
    cf.fit(sample_ratings_df)

    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    top_3 = cf.get_top_k(user_id=1, candidate_items=candidates, k=3)

    assert len(top_3) == 3
    # Check descending order
    assert top_3[0][1] >= top_3[1][1] >= top_3[2][1]
