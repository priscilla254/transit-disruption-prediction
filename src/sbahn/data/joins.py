"""Row-preserving joins of weather, disruptions, and line/station lookups.

Pandas only. The trip table is about 131,000 rows, small enough that each join
is a normal merge or a boolean mask. No PySpark or DuckDB.
"""

import pandas as pd


def _assert_trip_grain(before: pd.DataFrame, after: pd.DataFrame, step: str) -> None:
    if len(after) != len(before):
        raise AssertionError(
            f"{step} changed the row count from {len(before):,} to {len(after):,}"
        )
    if not after["trip_id"].is_unique:
        raise AssertionError(f"{step} duplicated trip_id")


def merge_weather(trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the weather row for the hour of scheduled departure."""
    out = trips.copy()
    out["_weather_hour"] = out["scheduled_departure_time"].dt.floor("h")
    weather_hour = weather.copy()
    weather_hour["_weather_hour"] = pd.to_datetime(weather_hour["timestamp"]).dt.floor("h")
    value_columns = [column for column in weather_hour.columns if column != "timestamp"]
    merged = out.merge(weather_hour[value_columns], on="_weather_hour", how="left")
    merged = merged.drop(columns=["_weather_hour"])
    _assert_trip_grain(trips, merged, "merge_weather")
    return merged


def flag_incidents(
    trips: pd.DataFrame, incidents: pd.DataFrame, window_minutes: int = 60
) -> pd.DataFrame:
    """Flag a trip when an incident on its line falls in the lookback window.

    delay_impact_factor is a generator weight, not a value known on the
    platform, so it is not selected. See docs/feature_dictionary.md.
    """
    incident_ids = (
        incidents["incident_id"]
        if "incident_id" in incidents.columns
        else pd.Series("", index=incidents.index)
    )
    usable = pd.DataFrame(
        {
            "line_id": incidents["line_id"],
            "incident_timestamp": pd.to_datetime(incidents["timestamp"]),
            "incident_type": incidents["incident_type"],
            "incident_id": incident_ids,
        }
    )
    left = trips.reset_index(drop=True)
    paired = left.merge(usable, on="line_id", how="left")
    window = pd.Timedelta(minutes=window_minutes)
    departure = paired["scheduled_departure_time"]
    occurred = paired["incident_timestamp"]
    in_window = occurred.notna() & (occurred <= departure) & (departure < occurred + window)
    # Latest timestamp in the window. A shared timestamp keeps the smallest incident_id.
    matched = paired.loc[in_window].sort_values(
        ["trip_id", "incident_timestamp", "incident_id"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    chosen = matched.drop_duplicates("trip_id", keep="last")
    flags = left[["trip_id"]].merge(
        chosen[["trip_id", "incident_type"]], on="trip_id", how="left"
    )
    out = left.copy()
    out["has_incident"] = flags["incident_type"].notna().astype("int64")
    out["incident_type"] = flags["incident_type"].to_numpy()
    _assert_trip_grain(trips, out, "flag_incidents")
    return out


def flag_events(trips: pd.DataFrame, city_events: pd.DataFrame) -> pd.DataFrame:
    """Flag a trip when its line is in the event's line list on the event date."""
    events = city_events.copy()
    events["line_list"] = (
        events["impact_on_lines"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(lambda parts: [part.strip() for part in parts if part.strip()])
    )
    long = events.explode("line_list")
    long["event_date"] = pd.to_datetime(long["date"]).dt.normalize()
    left = trips.reset_index(drop=True)
    left["_departure_date"] = left["scheduled_departure_time"].dt.normalize()
    paired = left.merge(
        long[["event_date", "line_list", "event_name"]],
        left_on=["_departure_date", "line_id"],
        right_on=["event_date", "line_list"],
        how="left",
    )
    # One event name per trip. Alphabetical event_name if two events match.
    paired = paired.sort_values(["trip_id", "event_name"], kind="mergesort")
    chosen = paired.drop_duplicates("trip_id", keep="first")
    out = chosen.drop(columns=["_departure_date", "event_date", "line_list"])
    out["is_event_day"] = out["event_name"].notna().astype("int64")
    out = out.set_index("trip_id").loc[left["trip_id"]].reset_index()
    _assert_trip_grain(trips, out, "flag_events")
    return out


def flag_strikes(trips: pd.DataFrame, strikes: pd.DataFrame) -> pd.DataFrame:
    """Flag a trip when departure is inside any half-open strike window."""
    out = trips.copy()
    departure = out["scheduled_departure_time"]
    in_strike = pd.Series(False, index=out.index)
    for start, end in zip(
        pd.to_datetime(strikes["strike_start"]),
        pd.to_datetime(strikes["strike_end"]),
        strict=True,
    ):
        in_strike = in_strike | ((departure >= start) & (departure < end))
    out["is_strike_day"] = in_strike.astype("int64")
    _assert_trip_grain(trips, out, "flag_strikes")
    return out


def merge_metadata(
    trips: pd.DataFrame, stations: pd.DataFrame, lines: pd.DataFrame
) -> pd.DataFrame:
    """Join the line and both stations. delay_propensity is left off the row.

    It is a line score that tracks the delay label, so joining it would hand
    the model a stand-in for y. See docs/feature_dictionary.md.
    """
    line_columns = [column for column in lines.columns if column != "delay_propensity"]
    out = trips.merge(lines[line_columns], on="line_id", how="left")
    station_columns = ["station_id", "name", "is_major_hub", "location_category"]
    start = stations[station_columns].rename(
        columns={
            "station_id": "start_station_id",
            "name": "start_name",
            "is_major_hub": "start_is_major_hub",
            "location_category": "start_location_category",
        }
    )
    end = stations[station_columns].rename(
        columns={
            "station_id": "end_station_id",
            "name": "end_name",
            "is_major_hub": "end_is_major_hub",
            "location_category": "end_location_category",
        }
    )
    out = out.merge(start, on="start_station_id", how="left")
    out = out.merge(end, on="end_station_id", how="left")
    _assert_trip_grain(trips, out, "merge_metadata")
    return out
