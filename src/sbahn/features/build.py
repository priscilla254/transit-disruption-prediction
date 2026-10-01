"""Derived columns for the joined trip table.

Pandas and numpy only. Lag rates use trips that already departed. Network load
counts the timetable for the same clock hour and does not use the delay label.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DROPPED_COLUMNS = (
    "delay_minutes",
    "is_cancelled",
    "delay_impact_factor",
    "delay_propensity",
)


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    departure = out["scheduled_departure_time"]
    out["hour"] = departure.dt.hour.astype("int64")
    out["day_of_week"] = departure.dt.dayofweek.astype("int64")
    out["month"] = departure.dt.month.astype("int64")
    out["is_weekend"] = departure.dt.dayofweek.ge(5).astype("int64")
    return out


def add_route(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["route_key"] = (
        out["line_id"].astype(str)
        + "|"
        + out["start_station_id"].astype(str)
        + "|"
        + out["end_station_id"].astype(str)
    )
    out["location_pair"] = (
        out["start_location_category"].astype(str)
        + " to "
        + out["end_location_category"].astype(str)
    )
    starts = out["start_is_major_hub"].astype(int).eq(1)
    ends = out["end_is_major_hub"].astype(int).eq(1)
    out["hub_pattern"] = np.select(
        [starts & ends, starts, ends],
        ["both", "start", "end"],
        default="neither",
    )
    return out


def add_weather_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_raining"] = out["precipitation_mm"].gt(0).astype("int64")
    out["rain_at_peak"] = (
        out["is_raining"].eq(1) & out["is_peak_hour"].astype(int).eq(1)
    ).astype("int64")
    return out


def add_network_load(frame: pd.DataFrame) -> pd.DataFrame:
    """Count scheduled trips in the same clock hour, including this trip."""
    out = frame.copy()
    hour_key = out["scheduled_departure_time"].dt.floor("h")
    out["network_load"] = hour_key.map(hour_key.value_counts()).astype("int64")
    return out


def _window_sums(sorted_times: np.ndarray, values: np.ndarray, window: np.timedelta64) -> tuple[np.ndarray, np.ndarray]:
    left = np.searchsorted(sorted_times, sorted_times - window, side="left")
    right = np.searchsorted(sorted_times, sorted_times, side="left")
    count = right - left
    cumulative = np.cumsum(values)
    total = np.zeros(len(sorted_times), dtype=np.float64)
    has_right = right > 0
    total[has_right] = cumulative[right[has_right] - 1]
    has_left = left > 0
    total[has_left] -= cumulative[left[has_left] - 1]
    return count, total


def add_prior_window(
    frame: pd.DataFrame,
    group_column: str,
    window: np.timedelta64,
    count_name: str,
    rate_name: str,
    mean_name: str | None = None,
) -> pd.DataFrame:
    """Aggregate earlier trips in a half-open window ending at this departure."""
    out = frame.copy()
    counts = pd.Series(0, index=out.index, dtype="int64")
    rates = pd.Series(np.nan, index=out.index, dtype="float64")
    means = pd.Series(np.nan, index=out.index, dtype="float64")
    for _, index in out.groupby(group_column, sort=False).groups.items():
        part = out.loc[index].sort_values("scheduled_departure_time", kind="mergesort")
        times = part["scheduled_departure_time"].to_numpy(dtype="datetime64[ns]")
        delayed = part["is_delayed"].to_numpy(dtype=np.float64)
        count, delayed_sum = _window_sums(times, delayed, window)
        observed = count > 0
        rate = np.full(len(part), np.nan)
        rate[observed] = delayed_sum[observed] / count[observed]
        counts.loc[part.index] = count
        rates.loc[part.index] = rate
        if mean_name is not None:
            minutes = part["delay_minutes"].to_numpy(dtype=np.float64)
            _, minute_sum = _window_sums(times, minutes, window)
            mean = np.full(len(part), np.nan)
            mean[observed] = minute_sum[observed] / count[observed]
            means.loc[part.index] = mean
    out[count_name] = counts
    out[rate_name] = rates
    if mean_name is not None:
        out[mean_name] = means
    return out


def add_lags(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_prior_window(
        frame,
        "line_id",
        np.timedelta64(1, "h"),
        "line_trips_prev_hour",
        "line_delay_rate_prev_hour",
        "line_mean_delay_prev_hour",
    )
    out = add_prior_window(
        out,
        "line_id",
        np.timedelta64(1, "D"),
        "line_trips_prev_day",
        "line_delay_rate_prev_day",
        "line_mean_delay_prev_day",
    )
    out = add_prior_window(
        out,
        "start_station_id",
        np.timedelta64(1, "h"),
        "start_station_trips_prev_hour",
        "start_station_delay_rate_prev_hour",
    )
    return out


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_lags(add_network_load(add_weather_flags(add_route(add_calendar(frame)))))
    dropped = [column for column in DROPPED_COLUMNS if column in out.columns]
    return out.drop(columns=dropped)


def write_features(in_path: Path, out_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(in_path)
    engineered = engineer(frame)
    if len(engineered) != len(frame):
        raise AssertionError(
            f"engineer changed the row count from {len(frame):,} to {len(engineered):,}"
        )
    if not engineered["trip_id"].is_unique:
        raise AssertionError("engineer duplicated trip_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(out_path, index=False)
    print(f"{len(engineered):,} rows")
    print("columns: " + ", ".join(engineered.columns))
    return engineered


def main() -> None:
    parser = argparse.ArgumentParser(description="Add engineered trip features.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_features(Path(args.in_path), Path(args.out))


if __name__ == "__main__":
    main()
