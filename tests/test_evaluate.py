import pytest
import pandas as pd
from src.evaluate import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_recommendations,
    full_evaluation_report,
)


def test_precision_at_k():
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 60}

    assert precision_at_k(recommended, relevant, k=5) == 2 / 5
    assert precision_at_k(recommended, relevant, k=2) == 1 / 2
    assert precision_at_k(recommended, relevant, k=1) == 0.0
    assert precision_at_k([], relevant, k=5) == 0.0


def test_recall_at_k():
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40}

    assert recall_at_k(recommended, relevant, k=5) == 2 / 2  # 1.0
    assert recall_at_k(recommended, relevant, k=2) == 1 / 2  # 0.5
    assert recall_at_k(recommended, set(), k=5) == 0.0


def test_ndcg_at_k():
    # Hit at rank 1
    assert ndcg_at_k([10, 20, 30], {10}, k=3) == 1.0

    # Hit at rank 2 vs rank 1: rank 1 NDCG must be strictly higher
    ndcg_rank1 = ndcg_at_k([10, 20, 30], {10}, k=3)
    ndcg_rank2 = ndcg_at_k([20, 10, 30], {10}, k=3)
    assert ndcg_rank1 > ndcg_rank2

    # No hits
    assert ndcg_at_k([20, 30, 40], {10}, k=3) == 0.0


def test_evaluate_recommendations():
    recommendations = {
        1: [10, 20, 30, 40, 50],
        2: [50, 60, 70, 80, 90],
    }
    test_df = pd.DataFrame([
        {"userId": 1, "movieId": 20, "rating": 5.0, "user_is_cold": False, "user_n_train": 10},
        {"userId": 2, "movieId": 99, "rating": 4.0, "user_is_cold": True, "user_n_train": 2},
    ])

    results_df = evaluate_recommendations(
        recommendations=recommendations,
        test_df=test_df,
        k_values=[5],
        relevance_threshold=4.0,
    )

    assert isinstance(results_df, pd.DataFrame)
    assert set(["userId", "k", "precision", "recall", "ndcg", "user_is_cold"]).issubset(results_df.columns)
    assert len(results_df) == 2

    # Test report aggregation
    report = full_evaluation_report(results_df, model_name="HybridTest")
    assert "model" in report.columns
    assert "slice" in report.columns
    assert "precision_mean" in report.columns
    assert "ndcg_mean" in report.columns
