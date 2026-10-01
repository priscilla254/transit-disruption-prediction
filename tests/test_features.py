import numpy as np
import pandas as pd

from sbahn.features.build import (
    add_calendar,
    add_lags,
    add_network_load,
    add_route,
    engineer,
)


def test_calendar_uses_monday_as_zero():
    frame = add_calendar(
        pd.DataFrame(
            {"scheduled_departure_time": pd.to_datetime(["2024-01-01 14:37:00"])}
        )
    )
    row = frame.iloc[0]
    assert row["hour"] == 14
    assert row["day_of_week"] == 0
    assert row["month"] == 1
    assert row["is_weekend"] == 0


def test_network_load_counts_the_scheduled_hour():
    departure = pd.to_datetime(
        [
            "2024-01-01 14:10:00",
            "2024-01-01 14:37:00",
            "2024-01-01 14:59:00",
            "2024-01-01 15:00:00",
        ]
    )
    on_time = add_network_load(
        pd.DataFrame(
            {
                "scheduled_departure_time": departure,
                "is_delayed": [0, 0, 0, 0],
            }
        )
    )
    delayed = add_network_load(
        pd.DataFrame(
            {
                "scheduled_departure_time": departure,
                "is_delayed": [1, 0, 0, 0],
            }
        )
    )
    assert on_time["network_load"].tolist() == [3, 3, 3, 1]
    assert delayed["network_load"].tolist() == [3, 3, 3, 1]


def test_route_and_hub_pattern():
    frame = add_route(
        pd.DataFrame(
            {
                "line_id": ["S1"],
                "start_station_id": ["S1"],
                "end_station_id": ["S2"],
                "start_location_category": ["central"],
                "end_location_category": ["suburb"],
                "start_is_major_hub": [1],
                "end_is_major_hub": [0],
            }
        )
    )
    assert frame.loc[0, "route_key"] == "S1|S1|S2"
    assert frame.loc[0, "location_pair"] == "central to suburb"
    assert frame.loc[0, "hub_pattern"] == "start"


def test_lags_use_only_strictly_earlier_trips():
    frame = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4, 5],
            "line_id": ["S1", "S1", "S1", "S1", "S2"],
            "start_station_id": ["A", "A", "A", "A", "A"],
            "scheduled_departure_time": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:30:00",
                    "2024-01-01 10:30:00",
                    "2024-01-01 10:45:00",
                    "2024-01-01 10:10:00",
                ]
            ),
            "is_delayed": [1, 0, 1, 1, 1],
            "delay_minutes": [10, 0, 30, 20, 40],
        }
    )
    lagged = add_lags(frame).set_index("trip_id")
    assert lagged.loc[2, "line_trips_prev_hour"] == 1
    assert lagged.loc[2, "line_delay_rate_prev_hour"] == 1.0
    assert lagged.loc[2, "line_mean_delay_prev_hour"] == 10.0
    assert lagged.loc[3, "line_trips_prev_hour"] == 1
    assert lagged.loc[3, "line_delay_rate_prev_hour"] == 1.0
    assert lagged.loc[1, "line_trips_prev_hour"] == 0
    assert np.isnan(lagged.loc[1, "line_delay_rate_prev_hour"])
    assert lagged.loc[4, "line_trips_prev_hour"] == 3
    assert lagged.loc[2, "start_station_trips_prev_hour"] == 2


def test_engineer_drops_delay_outcomes_and_proxies():
    frame = pd.DataFrame(
        {
            "trip_id": [1],
            "line_id": ["S1"],
            "start_station_id": ["S1"],
            "end_station_id": ["S2"],
            "scheduled_departure_time": pd.to_datetime(["2024-01-01 14:37:00"]),
            "is_peak_hour": [1],
            "is_delayed": [0],
            "delay_minutes": [2],
            "is_cancelled": [0],
            "delay_impact_factor": [1.5],
            "delay_propensity": [0.7],
            "precipitation_mm": [0.0],
            "start_is_major_hub": [0],
            "end_is_major_hub": [1],
            "start_location_category": ["suburb"],
            "end_location_category": ["central"],
        }
    )
    engineered = engineer(frame)
    assert "delay_minutes" not in engineered.columns
    assert "delay_propensity" not in engineered.columns
    assert "delay_impact_factor" not in engineered.columns
    assert "is_cancelled" not in engineered.columns
    assert engineered.loc[0, "is_delayed"] == 0
    assert engineered.loc[0, "hub_pattern"] == "end"

