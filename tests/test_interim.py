from pathlib import Path

import pandas as pd
import pytest

from sbahn.data.interim import cast_frame, write_interim


def test_cast_frame_parses_timestamps_and_dates():
    trips = cast_frame(
        "trips",
        pd.DataFrame({"scheduled_departure_time": ["2024-01-01 00:01:00"]}),
    )
    events = cast_frame("city_events", pd.DataFrame({"date": ["2024-09-06"]}))
    assert pd.api.types.is_datetime64_any_dtype(trips["scheduled_departure_time"])
    assert str(events["date"].dtype).startswith("date32")


def _valid_tables() -> dict[str, pd.DataFrame]:
    return {
        "trips": pd.DataFrame(
            {
                "trip_id": [1, 2],
                "line_id": ["S1", "S1"],
                "start_station_id": ["S1", "S2"],
                "end_station_id": ["S2", "S1"],
                "scheduled_departure_time": pd.to_datetime(
                    ["2024-01-01 00:01:00", "2024-01-01 00:02:00"]
                ),
            }
        ),
        "lines": pd.DataFrame({"line_id": ["S1"]}),
        "stations": pd.DataFrame({"station_id": ["S1", "S2"]}),
    }


def test_write_interim_refuses_orphan_line_ids(tmp_path: Path):
    tables = _valid_tables()
    tables["trips"].loc[0, "line_id"] = "S9"
    with pytest.raises(ValueError, match="orphan"):
        write_interim(tables, tmp_path)
    assert list(tmp_path.glob("*.parquet")) == []


def test_write_interim_writes_typed_parquet(tmp_path: Path):
    paths = write_interim(_valid_tables(), tmp_path)
    written = {path.name for path in paths}
    assert written == {"lines.parquet", "stations.parquet", "trips.parquet"}
    trips = pd.read_parquet(tmp_path / "trips.parquet")
    assert len(trips) == 2
    assert pd.api.types.is_datetime64_any_dtype(trips["scheduled_departure_time"])
