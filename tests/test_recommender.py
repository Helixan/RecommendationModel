import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender import EaseRecommender


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame(
            {"movie_id": list("abcdef"), "title": [f"Movie {letter}" for letter in "abcdef"]}
        )
        self.interactions = pd.DataFrame(
            [
                (1, "a", 6, 1), (1, "b", 5, 1), (1, "c", None, 1),
                (2, "a", 8, 1), (2, "d", 8, 1), (2, "e", 6, 1),
                (3, "b", 9, 1), (3, "d", 7, 1), (4, "a", 5, 1),
                (5, "e", 6, 1),
            ],
            columns=["user_id", "movie_id", "rating", "watch_count"],
        )

    def test_training_includes_moderate_ratings_and_unrated_watches(self):
        model = EaseRecommender()
        model.fit(self.interactions, self.movies)
        row = model.user_index[1]
        self.assertEqual(model.min_rating, 6)
        self.assertEqual(set(model.movie_ids[model.user_profiles[row]]), {"a", "c"})
        self.assertEqual(set(model.movie_ids[model.seen_movies[row]]), {"a", "b", "c"})
        self.assertTrue(model.has_positive_history(5))
        self.assertFalse(model.has_positive_history(4))
        self.assertFalse(model.has_positive_history(1001))
        recommended = set(model.recommend(1, 10)["movie_id"])
        self.assertTrue(recommended.isdisjoint({"a", "b", "c", "f"}))
        self.assertEqual(recommended, {"d", "e"})

    def test_weights_match_separate_ridge_regressions(self):
        model = EaseRecommender(regularization=2)
        model.fit(self.interactions, self.movies)
        matrix = model.user_profiles.astype(float)
        expected = np.zeros(model.weights.shape)

        for target in range(len(model.movie_ids)):
            columns = np.arange(len(model.movie_ids)) != target
            predictors = matrix[:, columns]
            gram = predictors.T @ predictors + 2 * np.eye(columns.sum())
            expected[columns, target] = np.linalg.solve(gram, predictors.T @ matrix[:, target])

        np.testing.assert_allclose(model.weights, expected, atol=1e-7)
        np.testing.assert_allclose(model.get_scores(1), matrix[0] @ expected, atol=1e-7)

    def test_saved_models_keep_their_training_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            for threshold in [6, 7]:
                with self.subTest(threshold=threshold):
                    model = EaseRecommender(regularization=25, min_rating=threshold)
                    model.fit(self.interactions, self.movies)
                    model.save(path)
                    loaded = EaseRecommender.load(path)
                    self.assertEqual(loaded.regularization, 25)
                    self.assertEqual(loaded.min_rating, threshold)
                    self.assertEqual(loaded.has_positive_history(5), threshold == 6)
                    np.testing.assert_array_equal(loaded.user_profiles, model.user_profiles)
                    pd.testing.assert_frame_equal(loaded.recommend(1), model.recommend(1))


if __name__ == "__main__":
    unittest.main()
