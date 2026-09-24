import argparse
from pathlib import Path
from time import perf_counter

from dataset import DATA_DIRECTORY, MovieDataset
from recommender import MODEL_PATH, EaseRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the movie recommendation model.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--regularization", type=float, default=100.0)
    parser.add_argument("--min-rating", type=int, default=7)
    arguments = parser.parse_args()

    try:
        model = EaseRecommender(arguments.regularization, arguments.min_rating)
        dataset = MovieDataset(arguments.data_dir)
        interactions = dataset.get_interactions()

        print(f"Training on {len(interactions):,} user-movie interactions...", flush=True)
        start = perf_counter()
        model.fit(interactions, dataset.movies)
        training_seconds = perf_counter() - start
        model.save(arguments.model_path)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Users with history: {len(model.user_ids):,}")
    print(f"Movies in catalog: {len(model.movie_ids):,}")
    print(f"Positive interactions: {model.user_profiles.sum():,}")
    print(f"Regularization: {model.regularization:g}")
    print(f"Minimum positive rating: {model.min_rating}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Model saved to {arguments.model_path.resolve()}")


if __name__ == "__main__":
    main()
