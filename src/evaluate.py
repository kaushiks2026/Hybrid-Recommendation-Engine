"""
Evaluation module: ranking metrics for recommendation quality.

Implements Precision@K, Recall@K, and NDCG@K with support for
sliced evaluation (cold vs. warm users).
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from src.utils import CONFIG, get_logger

logger = get_logger("evaluate")


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Precision@K: Of the top-K recommended items, how many are relevant?

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of truly relevant item IDs.
        k: Cutoff.

    Returns:
        Precision value in [0, 1].
    """
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Recall@K: Of all relevant items, how many appear in the top-K?

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of truly relevant item IDs.
        k: Cutoff.

    Returns:
        Recall value in [0, 1].
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Measures ranking quality — items at higher positions get more credit.
    A relevant item at position 1 is worth more than at position 10.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of truly relevant item IDs.
        k: Cutoff.

    Returns:
        NDCG value in [0, 1].
    """
    top_k = recommended[:k]

    # DCG: sum of 1/log2(rank+1) for each relevant hit
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 because ranks are 1-indexed

    # Ideal DCG: best possible ordering
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_recommendations(
    recommendations: dict[int, list[int]],
    test_df: pd.DataFrame,
    k_values: list[int] | None = None,
    relevance_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Evaluate recommendation quality across all users.

    Args:
        recommendations: Dict mapping userId → ordered list of recommended movieIds.
        test_df: Test DataFrame with columns [userId, movieId, rating] and
                 optional cold-start tags.
        k_values: List of K cutoffs (default: [5, 10, 20]).
        relevance_threshold: Minimum rating to consider an item "relevant".

    Returns:
        DataFrame with per-user metrics for each K.
    """
    k_values = k_values or CONFIG["top_k_values"]
    relevance_threshold = relevance_threshold or CONFIG["relevance_threshold"]

    logger.info(f"Evaluating with K={k_values}, relevance threshold={relevance_threshold}")

    # Build ground truth: for each user, the set of relevant test items
    # (items they rated >= threshold in the test set)
    user_relevant = {}
    for _, row in test_df.iterrows():
        uid = row["userId"]
        if row["rating"] >= relevance_threshold:
            if uid not in user_relevant:
                user_relevant[uid] = set()
            user_relevant[uid].add(row["movieId"])

    results = []
    for user_id, rec_list in recommendations.items():
        relevant = user_relevant.get(user_id, set())

        # Get cold-start info if available
        user_row = test_df[test_df["userId"] == user_id]
        user_is_cold = bool(user_row["user_is_cold"].iloc[0]) if "user_is_cold" in test_df.columns else False
        n_train = int(user_row["user_n_train"].iloc[0]) if "user_n_train" in test_df.columns else -1

        for k in k_values:
            results.append({
                "userId": user_id,
                "k": k,
                "precision": precision_at_k(rec_list, relevant, k),
                "recall": recall_at_k(rec_list, relevant, k),
                "ndcg": ndcg_at_k(rec_list, relevant, k),
                "user_is_cold": user_is_cold,
                "n_train": n_train,
                "n_relevant": len(relevant),
            })

    return pd.DataFrame(results)


def aggregate_metrics(
    metrics_df: pd.DataFrame,
    slice_name: str = "All",
) -> pd.DataFrame:
    """
    Aggregate per-user metrics into summary statistics.

    Args:
        metrics_df: Per-user metrics DataFrame from evaluate_recommendations.
        slice_name: Label for this slice (e.g., "All", "Cold", "Warm").

    Returns:
        Summary DataFrame with mean metrics per K.
    """
    summary = (
        metrics_df
        .groupby("k")
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            ndcg_mean=("ndcg", "mean"),
            n_users=("userId", "nunique"),
        )
        .reset_index()
    )
    summary["slice"] = slice_name
    return summary


def full_evaluation_report(
    metrics_df: pd.DataFrame,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Generate a complete evaluation report with All / Cold / Warm slices.

    Returns:
        Combined summary DataFrame.
    """
    # All users
    all_summary = aggregate_metrics(metrics_df, "All")

    # Cold users
    cold_df = metrics_df[metrics_df["user_is_cold"]]
    cold_summary = aggregate_metrics(cold_df, "Cold") if len(cold_df) > 0 else pd.DataFrame()

    # Warm users
    warm_df = metrics_df[~metrics_df["user_is_cold"]]
    warm_summary = aggregate_metrics(warm_df, "Warm") if len(warm_df) > 0 else pd.DataFrame()

    combined = pd.concat([all_summary, cold_summary, warm_summary], ignore_index=True)
    combined["model"] = model_name

    # Reorder columns for readability
    cols = ["model", "slice", "k", "precision_mean", "recall_mean", "ndcg_mean", "n_users"]
    combined = combined[cols]

    return combined


def print_report(report_df: pd.DataFrame) -> None:
    """Pretty-print an evaluation report."""
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)

    for model in report_df["model"].unique():
        model_df = report_df[report_df["model"] == model]
        print(f"\n[MODEL]: {model}")
        print("-" * 70)
        print(f"{'Slice':<8} {'K':>3}  {'P@K':>8}  {'R@K':>8}  {'NDCG@K':>8}  {'#Users':>7}")
        print("-" * 70)

        for _, row in model_df.iterrows():
            print(f"{row['slice']:<8} {int(row['k']):>3}  "
                  f"{row['precision_mean']:>8.4f}  "
                  f"{row['recall_mean']:>8.4f}  "
                  f"{row['ndcg_mean']:>8.4f}  "
                  f"{int(row['n_users']):>7}")

    print("=" * 80 + "\n")
