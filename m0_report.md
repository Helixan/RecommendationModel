# Milestone 0 - Movie Recommendation Model

## Learning

I used 54,704 events, 1,050 users, and 2,569 movies from the provided dataset. `events.csv.gz` supplies me with `timestamp`, `user_id`, `event_type`, `movie_id`, and `rating` from 1 to 10. `users.csv.gz` supplies me with `user_id`, `self_description_likes`, and `self_description_dislikes`. `movies.csv.gz` supplies me with `movie_id`, `title`, `genres`, and `overview`. Events link to users through `user_id` and movies through `movie_id`.

[Data preparation](src/dataset.py) groups watch and rating events by user/movie pair. It counts watches in `watch_count` and keeps the latest `rating`. `first_timestamp`/`last_timestamp` record time bounds for inspection only. This produces 27,327 pairs for 1,000 users, leaving 50 users without movie history. Ratings provide a clearer preference signal than watch events alone, while movie metadata and user descriptions help when that history is missing.

[EaseRecommender.fit](src/recommender.py), called by [train.py](src/train.py), trains EASE, a regularized linear recommender. Binary profiles mark ratings of at least 6 or unrated watches as positive. Learned movie-to-movie weights rank unseen movies from each user's positive history. I chose EASE because it learns shared preferences and trains quickly. This fits the task because users who enjoyed some of the same movies may also share preferences for movies they have not watched. Regularization helps reduce overfitting when each user has only a small amount of history. `seen_movies` includes all training pairs. Derived popularity is the fraction of training users with a positive interaction for a movie.

For users without positive history, [GPT6 Luna](src/preferences.py) converts their descriptions into liked and excluded genres, liked and disliked examples, and preference summaries. [Cold-start ranking](src/cold_start.py) matches normalized example titles to catalog IDs. It combines TF-IDF similarity from movie text and preference profiles, preferred-genre overlap, learned example relationships, and popularity. Movie text uses words and two-word phrases from titles, genres, and overviews. Disliked text reduces scores. Excluded genres, recognized examples, and seen movies are filtered out. Users with neither positive history nor descriptions get popular movies.

## Running your model

With Python 3.14 installed, run these commands in PowerShell.

```powershell
git clone https://github.com/Helixan/RecommendationModel.git
cd RecommendationModel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_data.py
python src/train.py
python src/recommend.py --user-id 1 --top-k 10
```

Replace `1` with the user ID. Training saves `models/recommender.npz`, and recommendations print as ranked JSON. If activation is blocked, use `.\.venv\Scripts\python.exe` for Python commands.

For cold start, add `OPENAI_TOKEN=your_api_key` to a root `.env` file using a key with `gpt-6-luna` access. Then run the following command.

```powershell
python src/recommend.py --user-id 1001 --top-k 10
```

Profiles are cached in `models/cold_start/`. Add `--offline` to require a cached profile. Training and users with positive history need no API key.
