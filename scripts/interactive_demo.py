"""
Interactive Recommender Demo & Testing Tool.

Allows querying recommendations for any user ID, simulating new cold-start users
with custom movie ratings, and viewing the adaptive blend weights and MMR re-ranking live.

Usage:
    python scripts/interactive_demo.py --user 42
    python scripts/interactive_demo.py --cold-demo
    python scripts/interactive_demo.py --custom-user "Toy Story (1995):5,Jurassic Park (1993):5"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all
from src.preprocessing import (
    train_test_split_leave_last_out,
    construct_cold_start_slice,
    tag_cold_start,
    get_user_item_sets,
)
from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter
from src.hybrid import HybridRecommender
from src.reranker import DiversityReranker
from src.utils import CONFIG, get_logger

logger = get_logger("demo")


def load_and_train_system():
    """Load data and train all models for interactive inference."""
    print("\n[INFO] Loading MovieLens dataset and initializing models...")
    ratings, movies, users = load_all()
    train_df, test_df = train_test_split_leave_last_out(ratings)
    train_df, test_df = construct_cold_start_slice(train_df, test_df, cold_user_ratio=0.20)
    test_df, _ = tag_cold_start(train_df, test_df)
    lookups = get_user_item_sets(train_df)

    print("[INFO] Training SVD Collaborative Filter...")
    cf_model = CollaborativeFilter()
    cf_model.fit(train_df)

    print("[INFO] Building Content-Based Model (TF-IDF)...")
    cb_model = ContentBasedFilter()
    cb_model.fit(movies, train_df)

    hybrid = HybridRecommender(cf_model, cb_model)
    hybrid.set_user_counts(lookups["user_counts"])

    reranker = DiversityReranker(cb_model, lam=0.7)
    movie_dict = movies.set_index("movieId")[["title", "genres"]].to_dict(orient="index")

    return {
        "movies": movies,
        "movie_dict": movie_dict,
        "train_df": train_df,
        "test_df": test_df,
        "lookups": lookups,
        "cf": cf_model,
        "cb": cb_model,
        "hybrid": hybrid,
        "reranker": reranker,
    }


def display_recommendations(user_id: int, sys_objs: dict, k: int = 10):
    """Display and compare recommendations for a given user ID."""
    lookups = sys_objs["lookups"]
    hybrid = sys_objs["hybrid"]
    cf = sys_objs["cf"]
    cb = sys_objs["cb"]
    reranker = sys_objs["reranker"]
    movie_dict = sys_objs["movie_dict"]

    n_train = lookups["user_counts"].get(user_id, 0)
    alpha = hybrid._get_alpha(user_id)
    seen = lookups["user_items"].get(user_id, set())
    candidates = [i for i in lookups["all_items"] if i not in seen]

    print("\n" + "=" * 80)
    print(f"RECOMMENDATION PROFILE FOR USER {user_id}")
    print("=" * 80)
    print(f"Training Interactions: {n_train} ratings")
    print(f"Cold-Start Status:     {'COLD (<5 ratings)' if n_train < 5 else 'WARM (>=5 ratings)'}")
    print(f"Adaptive Blend Weight: alpha = {alpha:.2f} (CF: {alpha*100:.1f}%, CB: {(1-alpha)*100:.1f}%)")
    print("-" * 80)

    # 1. Top CF recommendations
    cf_top = cf.get_top_k(user_id, candidates, k=k)
    print("\n[1] Top-5 Collaborative Filtering (SVD) Picks:")
    for rank, (mid, score) in enumerate(cf_top[:5], 1):
        m = movie_dict.get(mid, {"title": "Unknown", "genres": "N/A"})
        print(f"   {rank}. {m['title']} | {m['genres']} (Predicted Rating: {score:.2f})")

    # 2. Top CB recommendations
    cb_top = cb.get_top_k(user_id, candidates, k=k)
    print("\n[2] Top-5 Content-Based (TF-IDF) Picks:")
    for rank, (mid, score) in enumerate(cb_top[:5], 1):
        m = movie_dict.get(mid, {"title": "Unknown", "genres": "N/A"})
        print(f"   {rank}. {m['title']} | {m['genres']} (Cosine Sim: {score:.3f})")

    # 3. Hybrid recommendations
    hybrid_top = hybrid.get_top_k(user_id, candidates, k=k)
    print(f"\n[3] Top-{k} Blended Hybrid Recommendations:")
    for rank, (mid, score) in enumerate(hybrid_top, 1):
        m = movie_dict.get(mid, {"title": "Unknown", "genres": "N/A"})
        print(f"   {rank:2d}. {m['title']:<45} | {m['genres']:<25} (Score: {score:.3f})")

    # 4. MMR Re-ranking for diversity
    scored_candidates = hybrid.get_top_k(user_id, candidates, k=50)
    reranked = reranker.rerank(scored_candidates, k=k)
    ild_before = reranker.compute_ild([mid for mid, _ in hybrid_top])
    ild_after = reranker.compute_ild([mid for mid, _ in reranked])

    print(f"\n[4] Top-{k} Diversity Re-Ranked (MMR lambda=0.7):")
    for rank, (mid, score) in enumerate(reranked, 1):
        m = movie_dict.get(mid, {"title": "Unknown", "genres": "N/A"})
        print(f"   {rank:2d}. {m['title']:<45} | {m['genres']:<25}")

    print("\n[Diversity Metric]:")
    print(f"   Intra-List Diversity (ILD) Before MMR: {ild_before:.4f}")
    print(f"   Intra-List Diversity (ILD) After  MMR: {ild_after:.4f} (+{(ild_after - ild_before)*100:+.2f}%)")
    print("=" * 80 + "\n")


def run_cold_vs_warm_comparison(sys_objs: dict):
    """Demonstrate how the system shifts behavior between cold and warm users."""
    test_df = sys_objs["test_df"]
    lookups = sys_objs["lookups"]

    # Pick 1 cold user and 1 warm user
    cold_candidates = test_df[test_df["user_is_cold"]]["userId"].unique()
    warm_candidates = test_df[~test_df["user_is_cold"]]["userId"].unique()

    cold_user = int(cold_candidates[0])
    warm_user = int(warm_candidates[0])

    print("\n" + "#" * 80)
    print("DEMONSTRATION: COLD-START USER VS WARM-START USER")
    print("#" * 80)
    display_recommendations(cold_user, sys_objs, k=5)
    display_recommendations(warm_user, sys_objs, k=5)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommender Interactive Demo")
    parser.add_argument("--user", type=int, default=None, help="User ID to generate recommendations for (e.g. 1 to 6040)")
    parser.add_argument("--cold-demo", action="store_true", help="Compare cold-start vs warm-start users side-by-side")
    parser.add_argument("-k", type=int, default=10, help="Number of recommendations to show")

    args = parser.parse_args()

    sys_objs = load_and_train_system()

    if args.cold_demo or (args.user is None):
        run_cold_vs_warm_comparison(sys_objs)
    else:
        display_recommendations(args.user, sys_objs, k=args.k)


if __name__ == "__main__":
    main()
