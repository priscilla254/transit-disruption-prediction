"""Write validated, typed Parquet copies of the source tables."""

from pathlib import Path

import pandas as pd

from sbahn.config import INTERIM_DIR
from sbahn.data.load import connect, load_tables
from sbahn.data.validate import profile_tables, trip_reference_orphans

DATETIME_COLUMNS = {
    "incidents": ("timestamp",),
    "strikes": ("strike_start", "strike_end"),
    "trips": ("scheduled_departure_time",),
    "weather": ("timestamp",),
}
DATE_COLUMNS = {
    "city_events": ("date",),
}


def cast_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp text into datetime values and date text into dates."""
    out = frame.copy()
    for column in DATETIME_COLUMNS.get(name, ()):
        out[column] = pd.to_datetime(out[column])
    for column in DATE_COLUMNS.get(name, ()):
        out[column] = pd.to_datetime(out[column]).astype("date32[pyarrow]")
    return out


def cast_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: cast_frame(name, frame) for name, frame in tables.items()}


def validation_errors(tables: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    summary = profile_tables(tables)
    for row in summary.itertuples(index=False):
        if row.duplicate_rows:
            errors.append(f"{row.table} has {row.duplicate_rows} duplicate rows")
        if not row.key_unique:
            errors.append(f"{row.table}.{row.presumed_key} is not unique")
        if row.max_null_pct > 0:
            errors.append(f"{row.table} has nulls ({row.max_null_pct:.4f}% max)")
    if "trips" in tables:
        orphans = trip_reference_orphans(tables["trips"], tables["lines"], tables["stations"])
        for column, (count, missing) in orphans.items():
            if count:
                shown = ", ".join(missing)
                errors.append(f"trips.{column} has {count} orphan rows ({shown})")
    return errors


def write_interim(
    tables: dict[str, pd.DataFrame], directory: Path = INTERIM_DIR
) -> list[Path]:
    errors = validation_errors(tables)
    if errors:
        detail = "\n".join(errors)
        raise ValueError(f"Refusing to write interim Parquet:\n{detail}")
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in sorted(tables):
        path = directory / f"{name}.parquet"
        tables[name].to_parquet(path, index=False)
        paths.append(path)
    return paths


def main() -> None:
    with connect() as con:
        tables = load_tables(con)
    paths = write_interim(cast_tables(tables))
    for path in paths:
        frame = pd.read_parquet(path)
        print(f"{path.name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
