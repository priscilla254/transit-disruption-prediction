import sqlite3

import pandas as pd

from sbahn.config import database_path


def connect() -> sqlite3.Connection:
    return sqlite3.connect(database_path())


def load_tables(con: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    names = [
        row[0]
        for row in con.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
    ]
    return {name: pd.read_sql_query(f'SELECT * FROM "{name}"', con) for name in names}


# Timestamp columns are stored as text in SQLite. Parse them on the modeling path.
DATETIME_COLUMNS = {
    "city_events": ("date",),
    "incidents": ("timestamp",),
    "strikes": ("strike_start", "strike_end"),
    "trips": ("scheduled_departure_time",),
    "weather": ("timestamp",),
}


def load_all_tables(db_path: str) -> dict[str, pd.DataFrame]:
    """Load all seven tables, parse datetimes, and drop cancelled trips."""
    with sqlite3.connect(db_path) as con:
        tables = load_tables(con)
    for name, columns in DATETIME_COLUMNS.items():
        frame = tables[name]
        for column in columns:
            frame[column] = pd.to_datetime(frame[column])
    trips = tables["trips"]
    # Cancelled trips are a row filter, not a feature. See docs/feature_dictionary.md.
    cancelled = trips["is_cancelled"] != 0
    dropped = int(cancelled.sum())
    total = len(trips)
    percent = 100 * dropped / total if total else 0.0
    print(f"Dropped {dropped:,} cancelled trips ({percent:.2f}% of {total:,}).")
    tables["trips"] = trips.loc[~cancelled].reset_index(drop=True)
    return tables
