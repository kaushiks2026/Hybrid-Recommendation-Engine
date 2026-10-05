"""
Hybrid blending: adaptively combines Collaborative Filtering and
Content-Based scores based on how much interaction data exists for each user.

The blending weight α(u) is a function of the user's number of training
interactions:
    α(u) = min(n_ratings(u) / threshold, 1.0)

    hybrid_score = α · CF_score_norm + (1 - α) · CB_score_norm

This means:
  - Cold users (0 ratings) → α = 0 → 100% content-based
  - Users with some data  → α ∈ (0,1) → smooth blend
  - Warm users (≥threshold) → α = 1 → 100% collaborative
"""

import numpy as np
import pandas as pd

from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter
from src.utils import CONFIG, get_logger, Timer

logger = get_logger("hybrid")


class HybridRecommender:
    """
    Adaptive hybrid recommender that blends CF and CB scores.

    The blend weight shifts toward CF as the user's interaction count grows.
    """

    def __init__(
        self,
        cf_model: CollaborativeFilter,
        cb_model: ContentBasedFilter,
        blend_threshold: int | None = None,
    ):
        self.cf = cf_model
        self.cb = cb_model
        self.blend_threshold = blend_threshold or CONFIG["blend_threshold"]
        self.user_counts = {}  # userId → number of training interactions

    def set_user_counts(self, user_counts: dict[int, int]) -> None:
        """Set the training interaction counts for blending weight calculation."""
        self.user_counts = user_counts
        logger.info(f"Set interaction counts for {len(user_counts):,} users")

    def _get_alpha(self, user_id: int) -> float:
        """
        Compute the CF blending weight for a user.

        Returns value in [0, 1]:
          - 0 means 100% content-based (cold user)
          - 1 means 100% collaborative (warm user)
        """
        n = self.user_counts.get(user_id, 0)
        return min(n / self.blend_threshold, 1.0)

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        """
        Min-max normalize scores to [0, 1] range.

        This is crucial because CF outputs ratings (1-5) while CB outputs
        cosine similarities (0-1). Without normalization, CF would dominate
        simply due to its larger scale.
        """
        if not scores:
            return scores

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        rng = max_val - min_val

        if rng == 0:
            # All scores are identical → return uniform 0.5
            return {k: 0.5 for k in scores}

        return {k: (v - min_val) / rng for k, v in scores.items()}

    def score_items(
        self, user_id: int, candidate_items: list[int]
    ) -> dict[int, float]:
        """
        Compute blended scores for a list of candidate items.

        Returns:
            Dict mapping movieId → hybrid score.
        """
        alpha = self._get_alpha(user_id)

        # Get raw scores from both models
        cf_scores = self.cf.score_items(user_id, candidate_items)
        cb_scores = self.cb.score_items(user_id, candidate_items)

        # Normalize to [0, 1]
        cf_norm = self._normalize_scores(cf_scores)
        cb_norm = self._normalize_scores(cb_scores)

        # Blend
        hybrid_scores = {}
        for item_id in candidate_items:
            cf_val = cf_norm.get(item_id, 0.0)
            cb_val = cb_norm.get(item_id, 0.0)
            hybrid_scores[item_id] = alpha * cf_val + (1 - alpha) * cb_val

        return hybrid_scores

    def get_top_k(
        self,
        user_id: int,
        candidate_items: list[int],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get top-K recommended items for a user using hybrid scoring.

        Returns:
            List of (movieId, hybrid_score) tuples sorted descending.
        """
        scores = self.score_items(user_id, candidate_items)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def get_blend_stats(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Report blending weight distribution across test users.

        Returns a DataFrame summarizing alpha values.
        """
        alphas = []
        for user_id in test_df["userId"].unique():
            alpha = self._get_alpha(user_id)
            n = self.user_counts.get(user_id, 0)
            alphas.append({"userId": user_id, "n_train": n, "alpha": alpha})

        df = pd.DataFrame(alphas)
        logger.info(f"Alpha distribution: mean={df['alpha'].mean():.3f}, "
                    f"median={df['alpha'].median():.3f}, "
                    f"α=0 (pure CB): {(df['alpha'] == 0).sum()}, "
                    f"α=1 (pure CF): {(df['alpha'] >= 1.0).sum()}")
        return df
