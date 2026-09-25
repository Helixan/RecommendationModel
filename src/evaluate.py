import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from dataset import DATA_DIRECTORY, MovieDataset
from evaluation import InteractionSplitter, RankingEvaluator
from recommender import MODEL_PATH, EaseRecommender


def print_metrics(label: str, metrics: dict, top_k: int) -> None:
    print(
        f"{label}: "
        f"NDCG@{top_k}={metrics['ndcg_at_k']:.4f}, "
        f"Recall@{top_k}={metrics['recall_at_k']:.4f}, "
        f"Precision@{top_k}={metrics['precision_at_k']:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and tune the movie recommender.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=MODEL_PATH.parent / "evaluation.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rating", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--regularizations", type=float, nargs="+", default=[1, 10, 25, 50, 100, 250, 500]
    )
    arguments = parser.parse_args()

    try:
        splitter = InteractionSplitter(
            arguments.validation_fraction, arguments.test_fraction, arguments.seed
        )
        evaluator = RankingEvaluator(arguments.top_k, arguments.min_rating)
        candidates = sorted(set(arguments.regularizations))

        for regularization in candidates:
            EaseRecommender(regularization, arguments.min_rating)

        dataset = MovieDataset(arguments.data_dir)
        training, validation, test = splitter.split(dataset.get_interactions())
        print(
            f"Interactions: train={len(training):,}, "
            f"validation={len(validation):,}, test={len(test):,}",
            flush=True,
        )

        validation_results = []
        validation_popularity = None

        for regularization in candidates:
            model = EaseRecommender(regularization, arguments.min_rating)
            model.fit(training, dataset.movies)

            if validation_popularity is None:
                validation_popularity = evaluator.evaluate(model, validation, popularity_only=True)
                print_metrics("Validation popularity", validation_popularity, arguments.top_k)

            metrics = evaluator.evaluate(model, validation)
            validation_results.append({"regularization": regularization, "metrics": metrics})
            print_metrics(f"Validation EASE, regularization={regularization:g}", metrics, arguments.top_k)

        best = max(
            validation_results,
            key=lambda result: (
                result["metrics"]["ndcg_at_k"],
                result["metrics"]["recall_at_k"],
                result["regularization"],
            ),
        )
        selected_regularization = best["regularization"]
        print(f"Selected regularization: {selected_regularization:g}", flush=True)

        final_training = pd.concat([training, validation], ignore_index=True)
        final_model = EaseRecommender(selected_regularization, arguments.min_rating)
        final_model.fit(final_training, dataset.movies)
        test_popularity = evaluator.evaluate(final_model, test, popularity_only=True)
        test_model = evaluator.evaluate(final_model, test)
        print_metrics("Test popularity", test_popularity, arguments.top_k)
        print_metrics("Test EASE", test_model, arguments.top_k)

        file_hashes = {}
        for filename in ["events.csv.gz", "users.csv.gz", "movies.csv.gz"]:
            with (arguments.data_dir / filename).open("rb") as data_file:
                file_hashes[filename] = hashlib.file_digest(data_file, "sha256").hexdigest()

        result = {
            "split_method": "seeded random holdout within each user, grouped by user/movie",
            "seed": arguments.seed,
            "validation_fraction": arguments.validation_fraction,
            "test_fraction": arguments.test_fraction,
            "interaction_counts": {
                "training": len(training),
                "validation": len(validation),
                "test": len(test),
            },
            "cold_start_users_not_evaluated": len(dataset.get_cold_start_users()),
            "top_k": arguments.top_k,
            "relevance_rating": arguments.min_rating,
            "selection_rule": "validation NDCG, then recall, then higher regularization",
            "validation_popularity": validation_popularity,
            "validation_results": validation_results,
            "selected_regularization": selected_regularization,
            "test_training_data": "training and validation interactions",
            "test_popularity": test_popularity,
            "test_model": test_model,
            "data_sha256": file_hashes,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Evaluation saved to {arguments.output.resolve()}")
    print(f"Train on all interactions with --regularization {selected_regularization:g}")


if __name__ == "__main__":
    main()
