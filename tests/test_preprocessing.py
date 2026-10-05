import pandas as pd
import pytest

from src.preprocessing import (
    train_test_split_leave_last_out,
    construct_cold_start_slice,
    tag_cold_start,
    get_user_item_sets,
    build_popularity_ranking,
)


def test_train_test_split_leave_last_out(sample_ratings_df):
    train_df, test_df = train_test_split_leave_last_out(sample_ratings_df)

    # Every user in ratings should have exactly 1 rating in test
    assert len(test_df) == sample_ratings_df["userId"].nunique()
    assert test_df["userId"].nunique() == sample_ratings_df["userId"].nunique()

    # Total ratings should equal train + test
    assert len(train_df) + len(test_df) == len(sample_ratings_df)

    # Ensure test item corresponds to max timestamp for user 1
    user1_max_ts = sample_ratings_df[sample_ratings_df["userId"] == 1]["timestamp"].max()
    user1_test_ts = test_df[test_df["userId"] == 1]["timestamp"].iloc[0]
    assert user1_test_ts == user1_max_ts


def test_construct_cold_start_slice(sample_ratings_df):
    train_df, test_df = train_test_split_leave_last_out(sample_ratings_df)
    modified_train, test_out = construct_cold_start_slice(
        train_df, test_df, cold_user_ratio=0.5, max_cold_interactions=2, random_seed=42
    )

    assert len(test_out) == len(test_df)
    assert len(modified_train) <= len(train_df)
    assert modified_train["userId"].nunique() == train_df["userId"].nunique()


def test_tag_cold_start(sample_ratings_df):
    train_df, test_df = train_test_split_leave_last_out(sample_ratings_df)
    tagged_test, summary = tag_cold_start(train_df, test_df, threshold=4)

    assert "user_is_cold" in tagged_test.columns
    assert "item_is_cold" in tagged_test.columns
    assert "user_n_train" in tagged_test.columns
    assert summary["total_test_users"] == len(tagged_test)
    assert summary["cold_users"] + summary["warm_users"] == summary["total_test_users"]


def test_get_user_item_sets(sample_ratings_df):
    train_df, _ = train_test_split_leave_last_out(sample_ratings_df)
    lookups = get_user_item_sets(train_df)

    assert "user_items" in lookups
    assert "item_users" in lookups
    assert "all_items" in lookups
    assert "user_counts" in lookups
    assert isinstance(lookups["user_items"][1], set)


def test_build_popularity_ranking(sample_ratings_df):
    pop_ranked = build_popularity_ranking(sample_ratings_df)
    assert isinstance(pop_ranked, list)
    assert len(pop_ranked) == sample_ratings_df["movieId"].nunique()
