import sqlite3

import pandas as pd

from sbahn.config import database_path
from sbahn.data.load import load_tables

# These tables have no declared primary key. city_events and strikes have no id column.
PRESUMED_KEYS = {
    "city_events": "event_name",
    "incidents": "incident_id",
    "lines": "line_id",
    "stations": "station_id",
    "strikes": "strike_start",
    "trips": "trip_id",
    "weather": "timestamp",
}


def orphan_ids(values: pd.Series, parent: pd.Series) -> tuple[int, list[str]]:
    parent_ids = set(parent)
    missing = sorted(set(values) - parent_ids)
    orphan_rows = int((~values.isin(parent_ids)).sum())
    return orphan_rows, missing


def trip_reference_orphans(
    trips: pd.DataFrame, lines: pd.DataFrame, stations: pd.DataFrame
) -> dict[str, tuple[int, list[str]]]:
    return {
        "line_id": orphan_ids(trips["line_id"], lines["line_id"]),
        "start_station_id": orphan_ids(trips["start_station_id"], stations["station_id"]),
        "end_station_id": orphan_ids(trips["end_station_id"], stations["station_id"]),
    }


def profile_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in tables.items():
        key = PRESUMED_KEYS[name]
        null_pct = frame.isna().mean() * 100
        rows.append(
            {
                "table": name,
                "rows": len(frame),
                "duplicate_rows": int(frame.duplicated().sum()),
                "presumed_key": key,
                "key_unique": bool(frame[key].is_unique),
                "key_nunique": int(frame[key].nunique(dropna=False)),
                "max_null_pct": float(null_pct.max()),
            }
        )
    return pd.DataFrame(rows)


def print_schema(con: sqlite3.Connection) -> None:
    tables = load_tables(con)
    for name, frame in tables.items():
        declared = {
            col[1]: col[2] or ""
            for col in con.execute(f'PRAGMA table_info("{name}")')
        }
        print(f"\n{name}: {len(frame):,} rows")
        print(f"{'column':<32} {'sql_type':<16} pandas_dtype")
        for column in frame.columns:
            print(f"{column:<32} {declared.get(column, ''):<16} {frame[column].dtype}")


def main() -> None:
    with sqlite3.connect(database_path()) as con:
        tables = load_tables(con)
        print_schema(con)
    summary = profile_tables(tables)
    print("\nSummary")
    print(summary.to_string(index=False))
    orphans = trip_reference_orphans(tables["trips"], tables["lines"], tables["stations"])
    print(f"\ntrips: {len(tables['trips']):,} rows")
    for column, (count, missing) in orphans.items():
        print(f"{column} orphans: {count:,}")
        if count:
            print(f"  missing ids: {', '.join(missing)}")


if __name__ == "__main__":
    main()
