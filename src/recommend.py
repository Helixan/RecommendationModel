import argparse
import json
from pathlib import Path

from recommender import MODEL_PATH, EaseRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies for a user.")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    arguments = parser.parse_args()

    try:
        model = EaseRecommender.load(arguments.model_path)
        recommendations = model.recommend(arguments.user_id, arguments.top_k)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    method = "ease" if model.has_positive_history(arguments.user_id) else "popularity"
    result = {
        "user_id": arguments.user_id,
        "method": method,
        "recommendations": recommendations.to_dict(orient="records"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
