"""
Diversity Re-Ranker using Maximal Marginal Relevance (MMR).

After the hybrid model produces a scored list, the re-ranker re-orders it
to balance relevance with diversity — avoiding lists where all 10
recommendations are nearly identical (e.g., 10 action blockbusters).

MMR formula:
    MMR(item) = λ · relevance(item) - (1-λ) · max_sim(item, already_selected)

Also includes Intra-List Diversity (ILD) measurement.
"""

import numpy as np
from src.content_based import ContentBasedFilter
from src.utils import CONFIG, get_logger

logger = get_logger("reranker")


class DiversityReranker:
    """
    MMR-based re-ranker that promotes diversity in recommendation lists.

    Uses content similarity (from the CB model's TF-IDF vectors) to measure
    how similar a candidate is to items already in the re-ranked list.
    """

    def __init__(
        self,
        cb_model: ContentBasedFilter,
        lam: float | None = None,
    ):
        """
        Args:
            cb_model: Fitted ContentBasedFilter (used for item similarity).
            lam: Lambda parameter in [0, 1].
                 λ=1 → pure relevance (no diversity).
                 λ=0 → pure diversity (ignores relevance).
                 Default 0.7 → mostly relevant, some diversity.
        """
        self.cb = cb_model
        self.lam = lam or CONFIG["mmr_lambda"]

    def rerank(
        self,
        scored_items: list[tuple[int, float]],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Re-rank a scored list using MMR.

        Args:
            scored_items: List of (movieId, score) sorted by descending score.
            k: Number of items to return.

        Returns:
            Re-ranked list of (movieId, score) with improved diversity.
        """
        if len(scored_items) <= 1:
            return scored_items[:k]

        item_list = [item_id for item_id, _ in scored_items]
        n_cands = len(item_list)
        item_to_pos = {mid: i for i, mid in enumerate(item_list)}

        # Precompute candidate x candidate similarity matrix using sparse dot product
        valid_indices = [
            self.cb.item_id_to_idx.get(mid, -1) for mid in item_list
        ]
        # Build dense candidate vector matrix
        dim = self.cb.normalized_item_vectors.shape[1]
        dense_cand_vecs = np.zeros((n_cands, dim), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            if idx != -1:
                dense_cand_vecs[i] = self.cb.normalized_item_vectors[idx].toarray().flatten()

        sim_matrix = np.dot(dense_cand_vecs, dense_cand_vecs.T)

        # Normalize relevance scores to [0, 1]
        scores_dict = dict(scored_items)
        max_score = max(scores_dict.values())
        min_score = min(scores_dict.values())
        rng = max_score - min_score if max_score != min_score else 1.0
        norm_scores = {k: (v - min_score) / rng for k, v in scores_dict.items()}

        candidates = set(scores_dict.keys())
        selected = []

        # Greedily select items via MMR
        for _ in range(min(k, n_cands)):
            best_item = None
            best_mmr = -float("inf")

            for item_id in candidates:
                relevance = norm_scores[item_id]
                c_idx = item_to_pos[item_id]

                # Max similarity to already-selected items
                if selected:
                    max_sim = max(
                        sim_matrix[c_idx, item_to_pos[sel_id]]
                        for sel_id in selected
                    )
                else:
                    max_sim = 0.0

                mmr = self.lam * relevance - (1 - self.lam) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_item = item_id

            if best_item is not None:
                selected.append(best_item)
                candidates.discard(best_item)

        # Return with original scores
        return [(item_id, scores_dict[item_id]) for item_id in selected]

    def compute_ild(self, item_ids: list[int]) -> float:
        """
        Compute Intra-List Diversity (ILD): average pairwise dissimilarity.

        ILD = (1 / C(n,2)) * Σ (1 - sim(i, j)) for all pairs (i, j)

        Higher ILD = more diverse recommendations.

        Args:
            item_ids: List of recommended item IDs.

        Returns:
            ILD value in [0, 1]. Higher is more diverse.
        """
        if len(item_ids) < 2:
            return 0.0

        total_dissim = 0.0
        n_pairs = 0

        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                sim = self.cb.get_item_similarity(item_ids[i], item_ids[j])
                total_dissim += (1.0 - sim)
                n_pairs += 1

        return total_dissim / n_pairs if n_pairs > 0 else 0.0
