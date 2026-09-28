from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET = "alperenmyung/berlin-s-bahn-punctuality-database"
DB_NAME = "berlin_sbahn_delays.db"

# is_delayed is 1 exactly when delay_minutes is at least this many minutes.
DELAY_THRESHOLD_MINUTES = 6

KAGGLE_CACHE = Path.home() / (
    ".cache/kagglehub/datasets/alperenmyung/"
    "berlin-s-bahn-punctuality-database/versions/2"
)


def database_path() -> Path:
    local = RAW_DIR / DB_NAME
    if local.exists():
        return local
    return KAGGLE_CACHE / DB_NAME
