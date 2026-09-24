import argparse
from pathlib import Path

from dataset import DATA_DIRECTORY, MovieDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare the movie dataset.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    arguments = parser.parse_args()

    dataset = MovieDataset(arguments.data_dir)
    interactions = dataset.get_interactions()
    cold_start_users = dataset.get_cold_start_users()
    output_directory = arguments.data_dir / "processed"
    output_directory.mkdir(parents=True, exist_ok=True)

    interactions.to_csv(output_directory / "interactions.csv.gz", index=False)
    cold_start_users.to_csv(output_directory / "cold_start_users.csv.gz", index=False)

    print(f"Events: {len(dataset.events):,}")
    for event_type, count in dataset.events["event_type"].value_counts().items():
        print(f"  {event_type}: {count:,}")

    print(f"Users: {len(dataset.users):,}")
    print(f"Movies: {len(dataset.movies):,}")
    print(f"User-movie interactions: {len(interactions):,}")
    print(f"Users with interactions: {interactions['user_id'].nunique():,}")
    print(f"Cold-start users: {len(cold_start_users):,}")

    has_description = (
        cold_start_users["self_description_likes"].str.strip().ne("")
        | cold_start_users["self_description_dislikes"].str.strip().ne("")
    )
    print(f"Cold-start users with descriptions: {has_description.sum():,}")
    print(f"Prepared data saved to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
