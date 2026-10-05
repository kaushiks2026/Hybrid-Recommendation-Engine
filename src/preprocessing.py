"""
Preprocessing: train/test splitting, cold-start slicing, and data preparation.

Split strategy: **Leave-Last-Out** — for each user, the most recent rating
goes into the test set. This mimics real-world "predict the next item".
"""

import numpy as np
import pandas as pd

from src.utils import CONFIG, get_logger

logger = get_logger("preprocessing")


def train_test_split_leave_last_out(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings using leave-last-out: for each user, the most recent
    rating (by timestamp) becomes the test item; everything else is training.

    Returns:
        (train_df, test_df)
    """
    logger.info("Splitting data using leave-last-out strategy ...")

    # Sort by timestamp so we can pick the latest rating per user
    ratings_sorted = ratings.sort_values("timestamp")

    # For each user, the last row (most recent) is test
    test_idx = ratings_sorted.groupby("userId")["timestamp"].idxmax()
    test_df = ratings_sorted.loc[test_idx].copy()
    train_df = ratings_sorted.drop(test_idx).copy()

    logger.info(f"Train: {len(train_df):,} ratings | Test: {len(test_df):,} ratings")
    logger.info(f"Train users: {train_df['userId'].nunique():,} | "
                f"Test users: {test_df['userId'].nunique():,}")

    return train_df, test_df


def construct_cold_start_slice(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_user_ratio: float = 0.20,
    max_cold_interactions: int = 4,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explicitly construct a cold-start slice (<5 interactions) in training data.

    In the raw MovieLens 1M dataset, GroupLens filtered users to have >=20 ratings.
    To satisfy the assignment requirement of testing cold-start (<5 interactions),
    we randomly select a designated fraction of users (default 20%) and restrict
    their training history to a random count between 1 and max_cold_interactions (default 4).
    Their remaining training ratings are masked out, creating a true cold-start scenario.

    Returns:
        (modified_train_df, test_df)
    """
    logger.info(f"Explicitly constructing cold-start slice ({cold_user_ratio*100:.0f}% of users)...")
    rng = np.random.default_rng(random_seed)

    all_users = test_df["userId"].unique()
    n_cold = int(len(all_users) * cold_user_ratio)
    cold_users = set(rng.choice(all_users, size=n_cold, replace=False))

    # Mask training interactions for designated cold users
    keep_indices = []
    train_sorted = train_df.sort_values("timestamp")

    for user_id, group in train_sorted.groupby("userId"):
        if user_id in cold_users:
            # Retain only 1 to max_cold_interactions (e.g., 1-4) interactions
            n_keep = rng.integers(1, max_cold_interactions + 1)
            keep_indices.extend(group.index[-n_keep:])
        else:
            keep_indices.extend(group.index)

    modified_train = train_df.loc[keep_indices].copy()
    logger.info(f"Cold-start slice constructed: {len(cold_users):,} cold users, "
                f"retaining {len(modified_train):,} total training ratings.")

    return modified_train, test_df


def tag_cold_start(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Tag users and items as cold-start based on their training interaction count.

    A user/item is "cold" if it has fewer than `threshold` interactions in
    the training set.

    Returns:
        test_df with added 'user_is_cold' and 'item_is_cold' columns,
        and a summary dict with counts.
    """
    threshold = threshold or CONFIG["cold_start_threshold"]
    logger.info(f"Tagging cold-start with threshold < {threshold} interactions ...")

    # Count interactions per user and per item in TRAINING data
    user_counts = train_df.groupby("userId").size().to_dict()
    item_counts = train_df.groupby("movieId").size().to_dict()

    # Tag test users/items
    test_df = test_df.copy()
    test_df["user_n_train"] = test_df["userId"].map(user_counts).fillna(0).astype(int)
    test_df["item_n_train"] = test_df["movieId"].map(item_counts).fillna(0).astype(int)
    test_df["user_is_cold"] = test_df["user_n_train"] < threshold
    test_df["item_is_cold"] = test_df["item_n_train"] < threshold

    summary = {
        "total_test_users": len(test_df),
        "cold_users": int(test_df["user_is_cold"].sum()),
        "warm_users": int((~test_df["user_is_cold"]).sum()),
        "cold_items": int(test_df["item_is_cold"].sum()),
        "warm_items": int((~test_df["item_is_cold"]).sum()),
    }

    logger.info(f"Cold users: {summary['cold_users']:,} / {summary['total_test_users']:,} "
                f"({100*summary['cold_users']/summary['total_test_users']:.1f}%)")
    logger.info(f"Cold items: {summary['cold_items']:,} / {summary['total_test_users']:,} "
                f"({100*summary['cold_items']/summary['total_test_users']:.1f}%)")

    return test_df, summary


def get_user_item_sets(train_df: pd.DataFrame) -> dict:
    """
    Build lookup dictionaries for fast access during recommendation.

    Returns dict with:
      - 'user_items': {userId: set of movieIds they rated in training}
      - 'item_users': {movieId: set of userIds who rated it in training}
      - 'all_items': set of all movieIds in training
      - 'user_counts': {userId: number of training ratings}
    """
    logger.info("Building user-item lookup sets ...")

    user_items = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    item_users = train_df.groupby("movieId")["userId"].apply(set).to_dict()
    all_items = set(train_df["movieId"].unique())
    user_counts = train_df.groupby("userId").size().to_dict()

    return {
        "user_items": user_items,
        "item_users": item_users,
        "all_items": all_items,
        "user_counts": user_counts,
    }


def build_popularity_ranking(train_df: pd.DataFrame) -> list[int]:
    """
    Rank items by popularity (number of ratings) — used as a baseline.

    Returns:
        List of movieIds sorted by descending popularity.
    """
    popularity = train_df.groupby("movieId").size().sort_values(ascending=False)
    return popularity.index.tolist()
