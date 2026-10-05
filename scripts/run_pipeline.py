"""
End-to-end pipeline: data → train models → evaluate → failure analysis → diversity bonus.

Usage:
    python scripts/run_pipeline.py

This script runs the complete hybrid recommendation engine, producing:
  1. Trained CF (SVD) and CB (TF-IDF) models
  2. Hybrid blended recommendations
  3. Evaluation metrics (Precision@K, Recall@K, NDCG@K) for All/Cold/Warm slices
  4. Comparison against baselines (Popularity, CF-only, CB-only)
  5. Failure analysis
  6. (Bonus) Diversity re-ranking with MMR
"""

import sys
import json
import random
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all
from src.preprocessing import (
    train_test_split_leave_last_out,
    construct_cold_start_slice,
    tag_cold_start,
    get_user_item_sets,
    build_popularity_ranking,
)
from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter
from src.hybrid import HybridRecommender
from src.reranker import DiversityReranker
from src.evaluate import (
    evaluate_recommendations,
    full_evaluation_report,
    print_report,
)
from src.utils import CONFIG, RESULTS_DIR, get_logger, Timer, ensure_dirs

logger = get_logger("pipeline")

# Reproducibility
np.random.seed(CONFIG["random_seed"])
random.seed(CONFIG["random_seed"])


def generate_recommendations(
    model,
    test_users: list[int],
    lookups: dict,
    k: int = 20,
    model_name: str = "model",
) -> dict[int, list[int]]:
    """
    Generate top-K recommendations for all test users using a given model.

    For each user, the candidate items are all items NOT already rated
    in the training set.

    Args:
        model: Any model with a get_top_k(user_id, candidates, k) method.
        test_users: List of user IDs to generate recs for.
        lookups: Dict from get_user_item_sets().
        k: Number of recommendations per user.
        model_name: For logging.

    Returns:
        Dict mapping userId → ordered list of recommended movieIds.
    """
    logger.info(f"Generating top-{k} recs for {len(test_users):,} users [{model_name}] ...")
    recommendations = {}

    all_items = list(lookups["all_items"])

    for user_id in tqdm(test_users, desc=f"Recommending [{model_name}]", leave=False):
        # Candidate items = all items minus what user already rated in training
        seen = lookups["user_items"].get(user_id, set())
        candidates = [i for i in all_items if i not in seen]

        if not candidates:
            recommendations[user_id] = []
            continue

        top_k = model.get_top_k(user_id, candidates, k=k)
        recommendations[user_id] = [item_id for item_id, _ in top_k]

    return recommendations


def generate_popularity_recommendations(
    popularity_ranking: list[int],
    test_users: list[int],
    lookups: dict,
    k: int = 20,
) -> dict[int, list[int]]:
    """Generate recommendations based on global popularity (baseline)."""
    logger.info(f"Generating popularity baseline for {len(test_users):,} users ...")
    recommendations = {}

    for user_id in test_users:
        seen = lookups["user_items"].get(user_id, set())
        recs = [i for i in popularity_ranking if i not in seen][:k]
        recommendations[user_id] = recs

    return recommendations


def run_failure_analysis(
    hybrid_metrics: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    recommendations: dict[int, list[int]],
) -> str:
    """
    Analyze failure cases: where does the hybrid recommend poorly, and why?

    Returns a formatted string report.
    """
    logger.info("Running failure analysis ...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FAILURE ANALYSIS")
    report_lines.append("=" * 80)

    # Focus on K=10
    k10 = hybrid_metrics[hybrid_metrics["k"] == 10]

    # 1. Users with zero NDCG (complete misses)
    zero_ndcg_users = k10[k10["ndcg"] == 0.0]
    report_lines.append(f"\n[PATTERN 1]: Complete Misses (NDCG@10 = 0)")
    report_lines.append(f"   {len(zero_ndcg_users)} users ({100*len(zero_ndcg_users)/len(k10):.1f}% of test users)")

    if len(zero_ndcg_users) > 0:
        # Are they mostly cold users?
        cold_pct = zero_ndcg_users["user_is_cold"].mean() * 100
        report_lines.append(f"   Of these, {cold_pct:.1f}% are cold-start users")

        # What's the distribution of their training counts?
        avg_n = zero_ndcg_users["n_train"].mean()
        report_lines.append(f"   Average training interactions: {avg_n:.1f}")

    # 2. Analyze test items that were never recommended
    all_test_items = set(test_df["movieId"].unique())
    all_rec_items = set()
    for recs in recommendations.values():
        all_rec_items.update(recs)

    never_recommended = all_test_items - all_rec_items
    report_lines.append(f"\n[PATTERN 2]: Items Never Recommended")
    report_lines.append(f"   {len(never_recommended)} test items never appear in any user's top-20")

    if never_recommended and len(never_recommended) > 0:
        # Check if they're niche (few training ratings)
        item_counts = train_df.groupby("movieId").size()
        niche_items = [i for i in never_recommended if item_counts.get(i, 0) < 5]
        report_lines.append(f"   Of these, {len(niche_items)} have <5 training ratings (niche items)")

        # Show examples
        sample = list(never_recommended)[:5]
        for mid in sample:
            title = movies_df[movies_df["movieId"] == mid]["title"].values
            title_str = title[0] if len(title) > 0 else "Unknown"
            count = item_counts.get(mid, 0)
            report_lines.append(f"     -> '{title_str}' (only {count} training ratings)")

    # 3. Popularity bias analysis
    report_lines.append(f"\n[PATTERN 3]: Popularity Bias")
    item_popularity = train_df.groupby("movieId").size().sort_values(ascending=False)
    top_100_popular = set(item_popularity.head(100).index)

    rec_popularity_overlap = []
    for user_id, recs in recommendations.items():
        top10 = recs[:10]
        overlap = sum(1 for r in top10 if r in top_100_popular)
        rec_popularity_overlap.append(overlap / len(top10) if top10 else 0)

    avg_overlap = np.mean(rec_popularity_overlap)
    report_lines.append(f"   On average, {avg_overlap*100:.1f}% of top-10 recs are from the 100 most popular items")
    if avg_overlap > 0.5:
        report_lines.append("   [WARN] High popularity bias -- the model over-recommends blockbusters")
    else:
        report_lines.append("   [INFO] Moderate popularity spread")

    # 4. Genre monotonicity: users whose recs are all the same genre
    report_lines.append(f"\n[PATTERN 4]: Genre Monotonicity")
    movie_genres = movies_df.set_index("movieId")["genres"].to_dict()
    monotone_count = 0
    sample_monotone = []

    for user_id, recs in recommendations.items():
        top10 = recs[:10]
        if not top10:
            continue

        genres_in_recs = set()
        for mid in top10:
            g = movie_genres.get(mid, "")
            genres_in_recs.update(g.split("|"))

        if len(genres_in_recs) <= 2:
            monotone_count += 1
            if len(sample_monotone) < 3:
                sample_monotone.append((user_id, genres_in_recs))

    report_lines.append(f"   {monotone_count} users receive recommendations spanning <=2 genres")
    for uid, genres in sample_monotone:
        report_lines.append(f"     -> User {uid}: only genres {genres}")

    # 5. Contradictory taste users
    report_lines.append(f"\n📌 Pattern 5: Contradictory Taste Users")
    report_lines.append("   Users who rate very diverse genres highly confuse the content-based model,")
    report_lines.append("   which averages their taste vector into a blurred, unspecific profile.")

    # Find users who rated items across many genre categories
    user_genre_diversity = {}
    for user_id, group in train_df.groupby("userId"):
        high_rated = group[group["rating"] >= 4]
        all_genres = set()
        for mid in high_rated["movieId"]:
            g = movie_genres.get(mid, "")
            all_genres.update(g.split("|"))
        user_genre_diversity[user_id] = len(all_genres)

    diverse_users = {u: g for u, g in user_genre_diversity.items() if g >= 10}
    if diverse_users:
        diverse_with_metrics = k10[k10["userId"].isin(diverse_users.keys())]
        if len(diverse_with_metrics) > 0:
            avg_ndcg_diverse = diverse_with_metrics["ndcg"].mean()
            avg_ndcg_all = k10["ndcg"].mean()
            report_lines.append(f"   Users rating ≥10 genres highly: {len(diverse_users)}")
            report_lines.append(f"   Their avg NDCG@10: {avg_ndcg_diverse:.4f} vs overall: {avg_ndcg_all:.4f}")

    report_lines.append("\n" + "=" * 80)

    report_text = "\n".join(report_lines)
    print(report_text)
    return report_text


def save_results_plot(all_reports: pd.DataFrame) -> None:
    """Save a bar chart comparing models across slices."""
    ensure_dirs()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["precision_mean", "recall_mean", "ndcg_mean"]
    labels = ["Precision@10", "Recall@10", "NDCG@10"]

    k10 = all_reports[all_reports["k"] == 10]

    for ax, metric, label in zip(axes, metrics, labels):
        pivot = k10.pivot(index="slice", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Recommendation Quality: Model Comparison @ K=10", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {RESULTS_DIR / 'model_comparison.png'}")
    plt.close()


def save_alpha_distribution_plot(blend_stats: pd.DataFrame) -> None:
    """Save a histogram of blending weights."""
    ensure_dirs()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Alpha distribution
    axes[0].hist(blend_stats["alpha"], bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("α (CF weight)")
    axes[0].set_ylabel("Number of Users")
    axes[0].set_title("Blending Weight Distribution")
    axes[0].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="α=0.5")
    axes[0].legend()

    # Alpha vs training count
    axes[1].scatter(blend_stats["n_train"], blend_stats["alpha"],
                    alpha=0.3, s=5, color="#4C72B0")
    axes[1].set_xlabel("Number of Training Ratings")
    axes[1].set_ylabel("α (CF weight)")
    axes[1].set_title("Blending Weight vs. Interaction Count")

    plt.suptitle("Adaptive Blending Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "alpha_distribution.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved alpha distribution plot to {RESULTS_DIR / 'alpha_distribution.png'}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    """Run the complete hybrid recommendation pipeline."""
    ensure_dirs()

    print("\n" + "=" * 60)
    print("  HYBRID RECOMMENDATION ENGINE -- Full Pipeline")
    print("=" * 60 + "\n")

    # -- Phase 1: Data Loading & Preprocessing ------------------
    with Timer("Phase 1: Data loading & preprocessing", logger):
        ratings, movies, users = load_all()
        train_df, test_df = train_test_split_leave_last_out(ratings)
        train_df, test_df = construct_cold_start_slice(train_df, test_df, cold_user_ratio=0.20)
        test_df, cold_summary = tag_cold_start(train_df, test_df)
        lookups = get_user_item_sets(train_df)
        popularity_ranking = build_popularity_ranking(train_df)

    test_users = test_df["userId"].unique().tolist()
    max_k = max(CONFIG["top_k_values"])

    # ── Phase 2: Collaborative Filtering ───────────────────────
    with Timer("Phase 2: Collaborative Filtering (SVD)", logger):
        cf_model = CollaborativeFilter()
        cf_model.fit(train_df)

    # ── Phase 3: Content-Based Component ───────────────────────
    with Timer("Phase 3: Content-Based (TF-IDF)", logger):
        cb_model = ContentBasedFilter()
        cb_model.fit(movies, train_df)

    # ── Phase 4: Hybrid Blending ───────────────────────────────
    with Timer("Phase 4: Hybrid blending", logger):
        hybrid = HybridRecommender(cf_model, cb_model)
        hybrid.set_user_counts(lookups["user_counts"])
        blend_stats = hybrid.get_blend_stats(test_df)
        save_alpha_distribution_plot(blend_stats)

    # ── Phase 5: Generate Recommendations ──────────────────────
    with Timer("Phase 5: Generating recommendations", logger):
        # Popularity baseline
        pop_recs = generate_popularity_recommendations(
            popularity_ranking, test_users, lookups, k=max_k
        )

        # CF-only
        cf_recs = generate_recommendations(
            cf_model, test_users, lookups, k=max_k, model_name="CF-only"
        )

        # CB-only
        cb_recs = generate_recommendations(
            cb_model, test_users, lookups, k=max_k, model_name="CB-only"
        )

        # Hybrid
        hybrid_recs = generate_recommendations(
            hybrid, test_users, lookups, k=max_k, model_name="Hybrid"
        )

    # ── Phase 6: Evaluation ────────────────────────────────────
    with Timer("Phase 6: Evaluation", logger):
        all_reports = []

        for name, recs in [("Popularity", pop_recs), ("CF-only", cf_recs),
                           ("CB-only", cb_recs), ("Hybrid", hybrid_recs)]:
            metrics = evaluate_recommendations(recs, test_df)
            report = full_evaluation_report(metrics, model_name=name)
            all_reports.append(report)

        combined_report = pd.concat(all_reports, ignore_index=True)
        print_report(combined_report)

        # Save metrics to CSV
        combined_report.to_csv(RESULTS_DIR / "evaluation_metrics.csv", index=False)
        logger.info(f"Saved metrics to {RESULTS_DIR / 'evaluation_metrics.csv'}")

        # Save comparison plot
        save_results_plot(combined_report)

    # ── Phase 7: Failure Analysis ──────────────────────────────
    with Timer("Phase 7: Failure analysis", logger):
        hybrid_metrics = evaluate_recommendations(hybrid_recs, test_df)
        failure_report = run_failure_analysis(
            hybrid_metrics, test_df, train_df, movies, hybrid_recs
        )

        # Save failure report
        with open(RESULTS_DIR / "failure_analysis.txt", "w", encoding="utf-8") as f:
            f.write(failure_report)
        logger.info(f"Saved failure analysis to {RESULTS_DIR / 'failure_analysis.txt'}")

    # ── Phase 8 (Bonus): Diversity Re-Ranking ─────────────────
    with Timer("Phase 8: Diversity re-ranking (MMR)", logger):
        reranker = DiversityReranker(cb_model)

        # Re-rank hybrid recommendations
        reranked_recs = {}
        ild_before = []
        ild_after = []

        for user_id in tqdm(test_users, desc="Re-ranking", leave=False):
            # Get hybrid scored items
            seen = lookups["user_items"].get(user_id, set())
            candidates = [i for i in lookups["all_items"] if i not in seen]

            if not candidates:
                reranked_recs[user_id] = []
                continue

            scored = hybrid.get_top_k(user_id, candidates, k=50)

            # Measure ILD before re-ranking
            original_top10 = [item_id for item_id, _ in scored[:10]]
            ild_before.append(reranker.compute_ild(original_top10))

            # Re-rank
            reranked = reranker.rerank(scored, k=max_k)
            reranked_recs[user_id] = [item_id for item_id, _ in reranked]

            # Measure ILD after re-ranking
            reranked_top10 = [item_id for item_id, _ in reranked[:10]]
            ild_after.append(reranker.compute_ild(reranked_top10))

        # Evaluate re-ranked recommendations
        reranked_metrics = evaluate_recommendations(reranked_recs, test_df)
        reranked_report = full_evaluation_report(reranked_metrics, model_name="Hybrid+MMR")

        # Print comparison
        final_report = pd.concat([combined_report, reranked_report], ignore_index=True)
        print_report(final_report)

        # ILD comparison
        avg_ild_before = np.mean(ild_before)
        avg_ild_after = np.mean(ild_after)
        print(f"\n[METRICS] Intra-List Diversity (ILD) @ Top-10:")
        print(f"   Before MMR: {avg_ild_before:.4f}")
        print(f"   After  MMR: {avg_ild_after:.4f}")
        print(f"   Improvement: +{(avg_ild_after - avg_ild_before)*100:.2f}%")

        # Save final report
        final_report.to_csv(RESULTS_DIR / "final_evaluation.csv", index=False)
        logger.info(f"Saved final evaluation to {RESULTS_DIR / 'final_evaluation.csv'}")

        # Save ILD metrics
        ild_data = {
            "ild_before_mmr": float(avg_ild_before),
            "ild_after_mmr": float(avg_ild_after),
            "ild_improvement_pct": float((avg_ild_after - avg_ild_before) * 100),
        }
        with open(RESULTS_DIR / "diversity_metrics.json", "w") as f:
            json.dump(ild_data, f, indent=2)

    # -- Summary ------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n[OUTPUT] Results saved to: {RESULTS_DIR}")
    print("   * evaluation_metrics.csv -- All model metrics")
    print("   * final_evaluation.csv -- Including Hybrid+MMR")
    print("   * model_comparison.png -- Visual comparison chart")
    print("   * alpha_distribution.png -- Blending weight analysis")
    print("   * failure_analysis.txt -- Where the model still fails")
    print("   * diversity_metrics.json -- ILD before/after MMR")
    print()


if __name__ == "__main__":
    main()
