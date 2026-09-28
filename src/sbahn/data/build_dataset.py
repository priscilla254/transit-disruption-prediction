"""Build the joined trip table and write it to interim Parquet."""

import argparse
from pathlib import Path

from sbahn.config import DB_NAME, database_path
from sbahn.data.joins import (
    flag_events,
    flag_incidents,
    flag_strikes,
    merge_metadata,
    merge_weather,
)
from sbahn.data.load import load_all_tables
from sbahn.data.schemas import trips_schema


def resolve_database(raw_dir: Path) -> Path:
    local = raw_dir / DB_NAME
    if local.exists():
        return local
    fallback = database_path()
    print(f"{local} is missing. Using {fallback}.")
    return fallback


def build(db_path: Path, out_path: Path) -> None:
    print(f"Loading {db_path}.")
    tables = load_all_tables(str(db_path))
    trips = trips_schema.validate(tables["trips"])
    trips = merge_weather(trips, tables["weather"])
    trips = flag_incidents(trips, tables["incidents"])
    trips = flag_events(trips, tables["city_events"])
    trips = flag_strikes(trips, tables["strikes"])
    trips = merge_metadata(trips, tables["stations"], tables["lines"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(out_path, index=False)
    print(f"{len(trips):,} rows")
    print("columns: " + ", ".join(trips.columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Join source tables onto trips.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(resolve_database(Path(args.raw_dir)), Path(args.out))


if __name__ == "__main__":
    main()
