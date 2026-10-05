import pytest
from src.reranker import DiversityReranker


def test_diversity_reranker_rerank(fitted_models):
    reranker = fitted_models["reranker"]

    scored_items = [
        (3, 0.95),  # Comedy|Romance
        (7, 0.90),  # Comedy|Romance (very similar to item 3)
        (9, 0.85),  # Action (different genre)
        (5, 0.80),  # Comedy
    ]

    reranked = reranker.rerank(scored_items, k=3)
    assert len(reranked) == 3
    # Check that returned tuples maintain valid movieIds and scores
    reranked_ids = [mid for mid, _ in reranked]
    assert set(reranked_ids).issubset({3, 7, 9, 5})


def test_diversity_reranker_ild(fitted_models):
    reranker = fitted_models["reranker"]

    # Items with identical genres: 3 and 7 both have Comedy|Romance
    ild_low = reranker.compute_ild([3, 7])

    # Items with disjoint genres: 9 (Action) and 5 (Comedy)
    ild_high = reranker.compute_ild([9, 5])

    assert ild_high > ild_low
    assert 0.0 <= ild_low <= 1.0
    assert 0.0 <= ild_high <= 1.0


def test_diversity_reranker_edge_cases(fitted_models):
    reranker = fitted_models["reranker"]
    assert reranker.compute_ild([]) == 0.0
    assert reranker.compute_ild([1]) == 0.0
    assert reranker.rerank([], k=5) == []
