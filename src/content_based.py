"""
Content-Based component using TF-IDF on item metadata (genres).

Builds item feature vectors from genre tags, then constructs user taste
profiles by averaging the feature vectors of items they've rated (weighted
by rating). Works even for users with very few interactions.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import CONFIG, get_logger, Timer

logger = get_logger("content_based")


class ContentBasedFilter:
    """
    TF-IDF content-based recommender using item genre metadata.

    Each item gets a TF-IDF vector from its genres. Each user gets a
    "taste profile" vector that is the weighted average of their rated
    items' vectors.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            token_pattern=r"[A-Za-z'-]+",  # match genre words
            lowercase=True,
        )
        self.item_vectors = None          # sparse matrix: (n_items, n_features)
        self.item_id_to_idx = {}          # movieId → row index
        self.idx_to_item_id = {}          # row index → movieId
        self.user_profiles = {}           # userId → dense profile vector
        self._fitted = False

    def fit(
        self,
        movies_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> "ContentBasedFilter":
        """
        Build item TF-IDF vectors and user taste profiles.

        Args:
            movies_df: DataFrame with columns [movieId, title, genres]
                       genres are pipe-separated, e.g. "Action|Sci-Fi"
            train_df:  DataFrame with columns [userId, movieId, rating]
        """
        with Timer("Content-based model build", logger):
            self._build_item_vectors(movies_df)
            self._build_user_profiles(train_df)
            self._fitted = True

        return self

    def _build_item_vectors(self, movies_df: pd.DataFrame) -> None:
        """Convert genre strings to TF-IDF vectors."""
        logger.info("Building item TF-IDF vectors from genres ...")

        # Replace pipe separators with spaces so TF-IDF tokenizer works
        genre_docs = movies_df["genres"].str.replace("|", " ", regex=False)

        # Fit TF-IDF
        self.item_vectors = self.tfidf.fit_transform(genre_docs)
        from sklearn.preprocessing import normalize
        self.normalized_item_vectors = normalize(self.item_vectors, norm="l2", axis=1)

        # Build index mappings
        self.item_id_to_idx = {
            mid: idx for idx, mid in enumerate(movies_df["movieId"])
        }
        self.idx_to_item_id = {v: k for k, v in self.item_id_to_idx.items()}

        logger.info(f"Item vectors: {self.item_vectors.shape[0]} items x "
                    f"{self.item_vectors.shape[1]} features")
        logger.info(f"Features: {self.tfidf.get_feature_names_out()[:10]} ...")

    def _build_user_profiles(self, train_df: pd.DataFrame) -> None:
        """
        Build user taste profiles as rating-weighted averages of item vectors.

        Vectorized with sparse matrix multiplication: R @ item_vectors / weights.
        """
        logger.info("Building user taste profiles ...")

        valid_train = train_df[train_df["movieId"].isin(self.item_id_to_idx)]
        user_ids = valid_train["userId"].unique()
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        u_idx = valid_train["userId"].map(user_to_idx).values
        i_idx = valid_train["movieId"].map(self.item_id_to_idx).values
        ratings = valid_train["rating"].values.astype(np.float32)

        # User-item rating matrix: (n_users, n_items)
        n_items = self.item_vectors.shape[0]
        R = csr_matrix((ratings, (u_idx, i_idx)), shape=(len(user_ids), n_items))

        # Weight profiles by ratings: (n_users, n_items) @ (n_items, n_features) -> (n_users, n_features)
        weighted = R.dot(self.item_vectors).toarray()

        # Sum of ratings per user
        total_weights = np.array(R.sum(axis=1)).flatten()
        total_weights[total_weights == 0] = 1.0  # avoid divide by zero

        normalized = weighted / total_weights[:, np.newaxis]

        self.user_profiles = {uid: normalized[i] for i, uid in enumerate(user_ids)}
        logger.info(f"Built profiles for {len(self.user_profiles):,} users")

    def score_items(
        self, user_id: int, candidate_items: list[int]
    ) -> dict[int, float]:
        """
        Score candidate items for a user using cosine similarity between
        their taste profile and each item's TF-IDF vector.

        Vectorized with a single sparse dot product for speed.

        Returns:
            Dict mapping movieId -> similarity score.
        """
        profile = self.user_profiles.get(user_id)

        # Fallback for completely unknown users: return 0 scores
        if profile is None or np.allclose(profile, 0):
            return {item_id: 0.0 for item_id in candidate_items}

        norm = np.linalg.norm(profile)
        if norm == 0:
            return {item_id: 0.0 for item_id in candidate_items}

        profile_unit = (profile / norm).reshape(-1, 1)
        sims = self.normalized_item_vectors.dot(profile_unit).flatten()

        scores = {}
        for item_id in candidate_items:
            idx = self.item_id_to_idx.get(item_id)
            if idx is not None:
                scores[item_id] = float(sims[idx])
            else:
                scores[item_id] = 0.0

        return scores

    def get_top_k(
        self,
        user_id: int,
        candidate_items: list[int],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get top-K recommended items for a user.

        Returns:
            List of (movieId, score) tuples sorted by descending score.
        """
        scores = self.score_items(user_id, candidate_items)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def get_item_similarity(self, item_id_a: int, item_id_b: int) -> float:
        """Compute cosine similarity between two items."""
        if (item_id_a not in self.item_id_to_idx or
                item_id_b not in self.item_id_to_idx):
            return 0.0

        idx_a = self.item_id_to_idx[item_id_a]
        idx_b = self.item_id_to_idx[item_id_b]
        vec_a = self.normalized_item_vectors[idx_a]
        vec_b = self.normalized_item_vectors[idx_b]
        return float(vec_a.dot(vec_b.T)[0, 0])
