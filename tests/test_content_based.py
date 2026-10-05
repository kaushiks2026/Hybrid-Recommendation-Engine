import pytest
from src.content_based import ContentBasedFilter


def test_content_based_fit_and_profiles(sample_movies_df, sample_ratings_df):
    cb = ContentBasedFilter()
    cb.fit(sample_movies_df, sample_ratings_df)

    assert cb._fitted is True
    assert cb.item_vectors is not None
    assert len(cb.user_profiles) == sample_ratings_df["userId"].nunique()


def test_content_based_score_items(sample_movies_df, sample_ratings_df):
    cb = ContentBasedFilter()
    cb.fit(sample_movies_df, sample_ratings_df)

    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scores = cb.score_items(user_id=1, candidate_items=candidates)

    assert len(scores) == len(candidates)
    for mid, score in scores.items():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0001  # cosine similarity range


def test_content_based_unknown_user(sample_movies_df, sample_ratings_df):
    cb = ContentBasedFilter()
    cb.fit(sample_movies_df, sample_ratings_df)

    candidates = [1, 2, 3]
    scores = cb.score_items(user_id=9999, candidate_items=candidates)
    assert all(s == 0.0 for s in scores.values())


def test_item_similarity(sample_movies_df, sample_ratings_df):
    cb = ContentBasedFilter()
    cb.fit(sample_movies_df, sample_ratings_df)

    # Item 3 and 7 both have genres 'Comedy|Romance' -> similarity should be close to 1.0
    sim_identical_genre = cb.get_item_similarity(3, 7)
    assert pytest.approx(sim_identical_genre, rel=1e-3) == 1.0

    # Item 9 (Action) vs Item 5 (Comedy) -> disjoint genres -> similarity should be 0.0
    sim_disjoint = cb.get_item_similarity(9, 5)
    assert pytest.approx(sim_disjoint, abs=1e-5) == 0.0
