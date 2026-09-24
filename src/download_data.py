import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
DATA_URL = (
    "https://media.githubusercontent.com/media/mlip-cmu-online/public-data/"
    "0687388e6d91f0c43ba30a3b80facb7261341eff/m0/data"
)
FILE_HASHES = {
    "events.csv.gz": "5c61b29f1eab57ae0ff122062390e312f6813c4751eaab466997cdb55b712efb",
    "users.csv.gz": "7a7b2afbb598753af89cad7d865089d03dcaf0c834b2d42275f5918b97343809",
    "movies.csv.gz": "1e0dd97dccaff8cfe65d888cbdaec33e4a2673880818fbc3dcf859dc7806b147",
}


class DatasetDownloader:
    def __init__(self, data_directory: Path):
        self.data_directory = data_directory

    def download(self) -> None:
        self.data_directory.mkdir(parents=True, exist_ok=True)

        for filename, expected_hash in FILE_HASHES.items():
            destination = self.data_directory / filename

            if destination.exists() and self.get_checksum(destination) == expected_hash:
                print(f"Verified {filename}")
                continue

            temporary_path = destination.with_suffix(destination.suffix + ".part")

            try:
                with urlopen(f"{DATA_URL}/{filename}", timeout=60) as response:
                    with temporary_path.open("wb") as output:
                        shutil.copyfileobj(response, output)

                if self.get_checksum(temporary_path) != expected_hash:
                    raise ValueError(f"Checksum does not match for {filename}")

                temporary_path.replace(destination)
                print(f"Downloaded and verified {filename}")
            finally:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def get_checksum(path: Path) -> str:
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Milestone 0 dataset.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    arguments = parser.parse_args()

    downloader = DatasetDownloader(arguments.data_dir)
    downloader.download()


if __name__ == "__main__":
    main()
