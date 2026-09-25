from pathlib import Path

import pandas as pd


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


class MovieDataset:
    """Load the course data and combine watch and rating events."""

    def __init__(self, data_directory: Path = DATA_DIRECTORY):
        self.data_directory = data_directory
        self.events = self._read_table(
            "events.csv.gz",
            ["timestamp", "user_id", "event_type", "movie_id", "rating"],
        )
        self.users = self._read_table(
            "users.csv.gz",
            ["user_id", "self_description_likes", "self_description_dislikes"],
        )
        self.movies = self._read_table(
            "movies.csv.gz",
            ["movie_id", "title", "genres", "overview", "runtime", "license_cost"],
        )

        self.events["user_id"] = self.events["user_id"].astype("int64")
        self.users["user_id"] = self.users["user_id"].astype("int64")
        self.events["timestamp"] = pd.to_datetime(
            self.events["timestamp"], format="ISO8601", errors="raise", utc=True
        )
        self.events["rating"] = pd.to_numeric(
            self.events["rating"].replace("", pd.NA), errors="raise"
        ).astype("Int64")

        self._validate()

    def _read_table(self, filename: str, required_columns: list[str]) -> pd.DataFrame:
        path = self.data_directory / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run python src/download_data.py first."
            )

        # Keep empty descriptions as strings
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing_columns = set(required_columns) - set(table.columns)

        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"{filename} is missing columns: {columns}")

        if table.empty:
            raise ValueError(f"{filename} contains no rows")

        return table

    def _validate(self) -> None:
        if self.users["user_id"].duplicated().any():
            raise ValueError("User IDs must be unique")

        if (self.users["user_id"] <= 0).any():
            raise ValueError("User IDs must be positive")

        if self.movies["movie_id"].eq("").any():
            raise ValueError("Movie IDs cannot be empty")

        if self.movies["movie_id"].duplicated().any():
            raise ValueError("Movie IDs must be unique")

        if self.events["timestamp"].isna().any():
            raise ValueError("Events must have timestamps")

        valid_types = ["watch", "rating", "account_created"]

        if not self.events["event_type"].isin(valid_types).all():
            raise ValueError("Events contain an unsupported event type")

        if not self.events["user_id"].isin(self.users["user_id"]).all():
            raise ValueError("Events refer to users missing from users.csv.gz")

        movie_events = self.events["event_type"].isin(["watch", "rating"])

        if not self.events.loc[movie_events, "movie_id"].isin(self.movies["movie_id"]).all():
            raise ValueError("Events refer to movies missing from movies.csv.gz")

        rating_events = self.events["event_type"].eq("rating")
        ratings = self.events.loc[rating_events, "rating"]

        if ratings.isna().any() or not ratings.between(1, 10).all():
            raise ValueError("Ratings must be integers between 1 and 10")

        if self.events.loc[~rating_events, "rating"].notna().any():
            raise ValueError("Only rating events may contain ratings")

        if self.events.loc[~movie_events, "movie_id"].ne("").any():
            raise ValueError("Account creation events cannot contain movies")

    def get_interactions(self, events: pd.DataFrame | None = None) -> pd.DataFrame:
        if events is None:
            events = self.events

        movie_events = events.loc[
            events["event_type"].isin(["watch", "rating"])
        ].copy()
        movie_events["watch_count"] = movie_events["event_type"].eq("watch").astype("int64")
        keys = ["user_id", "movie_id"]

        interactions = movie_events.groupby(keys, as_index=False).agg(
            watch_count=("watch_count", "sum"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )

        ratings = movie_events.loc[movie_events["event_type"].eq("rating")]
        ratings = ratings.sort_values("timestamp", kind="stable")
        # Keep the latest rating if a user rates the same movie again
        ratings = ratings.drop_duplicates(keys, keep="last")

        interactions = interactions.merge(
            ratings[keys + ["rating"]], on=keys, how="left", validate="one_to_one"
        )
        interactions = interactions.sort_values(keys).reset_index(drop=True)

        return interactions[
            keys + ["watch_count", "rating", "first_timestamp", "last_timestamp"]
        ]

    def get_cold_start_users(self) -> pd.DataFrame:
        movie_events = self.events["event_type"].isin(["watch", "rating"])
        active_users = self.events.loc[movie_events, "user_id"]
        cold_start_users = self.users.loc[~self.users["user_id"].isin(active_users)]

        return cold_start_users.sort_values("user_id").reset_index(drop=True)
