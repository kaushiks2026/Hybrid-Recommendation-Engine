"""
Collaborative Filtering component using SVD (matrix factorization).

Uses the Surprise library's SVD algorithm which implements Simon Funk-style
SVD with user/item biases and regularized stochastic gradient descent.
"""

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

from src.utils import CONFIG, get_logger, Timer

logger = get_logger("collaborative")


class CollaborativeFilter:
    """
    SVD-based collaborative filtering model.

    Learns latent factor representations for users and items from the
    rating matrix. For any (user, item) pair it can predict a rating.
    """

    def __init__(
        self,
        n_factors: int | None = None,
        n_epochs: int | None = None,
        lr_all: float | None = None,
        reg_all: float | None = None,
        random_state: int | None = None,
    ):
        self.n_factors = n_factors or CONFIG["svd_n_factors"]
        self.n_epochs = n_epochs or CONFIG["svd_n_epochs"]
        self.lr_all = lr_all or CONFIG["svd_lr_all"]
        self.reg_all = reg_all or CONFIG["svd_reg_all"]
        self.random_state = random_state or CONFIG["random_seed"]

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        self._global_mean = None
        self._trained = False

    def fit(self, train_df: pd.DataFrame) -> "CollaborativeFilter":
        """
        Train the SVD model on the training ratings.

        Args:
            train_df: DataFrame with columns [userId, movieId, rating]
        """
        with Timer("SVD training", logger):
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                train_df[["userId", "movieId", "rating"]], reader
            )
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            self._global_mean = trainset.global_mean
            self._trained = True

        logger.info(f"SVD trained with {self.n_factors} factors, "
                    f"{self.n_epochs} epochs")
        logger.info(f"Global mean rating: {self._global_mean:.3f}")
        return self

    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Predict a rating for a single (user, item) pair.

        For unknown users/items, Surprise falls back to global mean + biases
        where available.
        """
        prediction = self.model.predict(uid=user_id, iid=item_id)
        return prediction.est

    def score_items(
        self, user_id: int, candidate_items: list[int]
    ) -> dict[int, float]:
        """
        Score a list of candidate items for a given user using fast vectorization.

        Returns:
            Dict mapping movieId → predicted rating.
        """
        if not self._trained:
            return {item_id: self.global_mean for item_id in candidate_items}

        trainset = self.model.trainset
        scores = {}

        # Check if user is in training set
        user_known = trainset.knows_user(user_id) if hasattr(trainset, "knows_user") else False
        if not user_known and hasattr(trainset, "_raw2inner_id_users"):
            user_known = user_id in trainset._raw2inner_id_users

        if user_known:
            u = trainset.to_inner_uid(user_id)
            u_bias = float(self.model.bu[u])
            u_vec = self.model.pu[u]
        else:
            u_bias = 0.0
            u_vec = None

        g_mean = float(self._global_mean)

        for item_id in candidate_items:
            item_known = item_id in trainset._raw2inner_id_items if hasattr(trainset, "_raw2inner_id_items") else False
            if item_known:
                i = trainset.to_inner_iid(item_id)
                i_bias = float(self.model.bi[i])
                if u_vec is not None:
                    est = g_mean + u_bias + i_bias + float(np.dot(self.model.qi[i], u_vec))
                else:
                    est = g_mean + i_bias
            else:
                est = g_mean + u_bias

            # Clip to valid rating range [1, 5]
            scores[item_id] = min(5.0, max(1.0, est))

        return scores

    def get_top_k(
        self,
        user_id: int,
        candidate_items: list[int],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get top-K recommended items for a user from candidate set.

        Returns:
            List of (movieId, score) tuples sorted by descending score.
        """
        scores = self.score_items(user_id, candidate_items)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    @property
    def global_mean(self) -> float:
        """Global mean rating from training data."""
        return self._global_mean if self._global_mean else 3.0
