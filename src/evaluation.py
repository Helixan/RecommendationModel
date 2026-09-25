import numpy as np
import pandas as pd

from recommender import EaseRecommender


class InteractionSplitter:
    def __init__(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 42,
    ):
        if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
            raise ValueError("Validation and test fractions must be between 0 and 1")

        if validation_fraction + test_fraction >= 1:
            raise ValueError("Validation and test fractions must leave training data")

        if seed < 0:
            raise ValueError("The random seed cannot be negative")

        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.seed = seed

    def split(self, interactions: pd.DataFrame) -> tuple:
        if interactions.empty:
            raise ValueError("Cannot split an empty interaction table")

        keys = ["user_id", "movie_id"]

        if interactions.duplicated(keys).any():
            raise ValueError("Split requires one row per user and movie")

        ordered = interactions.sort_values(keys).reset_index(drop=True)
        random = np.random.default_rng(self.seed)
        training_indices = []
        validation_indices = []
        test_indices = []

        for user_id, history in ordered.groupby("user_id", sort=True):
            count = len(history)

            if count < 3:
                training_indices.extend(history.index)
                continue

            indices = random.permutation(history.index.to_numpy())
            validation_count = min(max(1, int(count * self.validation_fraction)), count - 2)
            test_count = min(
                max(1, int(count * self.test_fraction)), count - validation_count - 1
            )
            validation_end = validation_count
            test_end = validation_count + test_count

            validation_indices.extend(indices[:validation_end])
            test_indices.extend(indices[validation_end:test_end])
            training_indices.extend(indices[test_end:])

        training = ordered.loc[sorted(training_indices)].reset_index(drop=True)
        validation = ordered.loc[sorted(validation_indices)].reset_index(drop=True)
        test = ordered.loc[sorted(test_indices)].reset_index(drop=True)

        return training, validation, test


class RankingEvaluator:
    def __init__(self, top_k: int = 10, relevance_rating: int = 7):
        if top_k <= 0:
            raise ValueError("The number of recommendations must be positive")

        if relevance_rating < 1 or relevance_rating > 10:
            raise ValueError("Relevance rating must be between 1 and 10")

        self.top_k = top_k
        self.relevance_rating = relevance_rating

    def evaluate(
        self,
        model: EaseRecommender,
        held_out: pd.DataFrame,
        popularity_only: bool = False,
    ) -> dict:
        positives = held_out.loc[held_out["rating"].ge(self.relevance_rating).fillna(False)]
        relevant_movies = positives.groupby("user_id")["movie_id"].agg(set)

        if relevant_movies.empty:
            raise ValueError("The held-out data has no relevant movie ratings")

        precision_values = []
        recall_values = []
        ndcg_values = []
        hit_values = []
        recommended_movies = set()
        supported_movies = set(model.movie_ids[model.popularity > 0])
        relevant_count = 0
        supported_count = 0

        for user_id, targets in relevant_movies.items():
            recommendations = model.recommend(
                int(user_id), self.top_k, popularity_only=popularity_only
            )
            movie_ids = recommendations["movie_id"].tolist()
            hits = np.array([movie_id in targets for movie_id in movie_ids], dtype=float)
            hit_count = int(hits.sum())
            discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
            ideal_count = min(len(targets), self.top_k)
            ideal_dcg = (1.0 / np.log2(np.arange(2, ideal_count + 2))).sum()

            precision_values.append(hit_count / self.top_k)
            recall_values.append(hit_count / len(targets))
            ndcg_values.append(float((hits * discounts).sum() / ideal_dcg))
            hit_values.append(float(hit_count > 0))
            recommended_movies.update(movie_ids)
            relevant_count += len(targets)
            supported_count += len(targets & supported_movies)

        return {
            "evaluated_users": len(relevant_movies),
            "users_without_relevant_ratings": int(
                held_out["user_id"].nunique() - len(relevant_movies)
            ),
            "relevant_interactions": relevant_count,
            "precision_at_k": float(np.mean(precision_values)),
            "recall_at_k": float(np.mean(recall_values)),
            "ndcg_at_k": float(np.mean(ndcg_values)),
            "hit_rate_at_k": float(np.mean(hit_values)),
            "catalog_coverage": len(recommended_movies) / len(model.movie_ids),
            "supported_target_fraction": supported_count / relevant_count,
        }
