import pandas as pd

from sbahn.data.joins import (
    flag_events,
    flag_incidents,
    flag_strikes,
    merge_metadata,
    merge_weather,
)
from sbahn.data.validate import orphan_ids, trip_reference_orphans


def test_orphan_ids_counts_missing_parents():
    values = pd.Series(["S1", "S9", "S1"])
    parent = pd.Series(["S1", "S2"])
    count, missing = orphan_ids(values, parent)
    assert count == 1
    assert missing == ["S9"]


def test_trip_references_match_when_ids_exist():
    trips = pd.DataFrame(
        {
            "line_id": ["S1", "S41"],
            "start_station_id": ["S1", "S2"],
            "end_station_id": ["S3", "S1"],
        }
    )
    lines = pd.DataFrame({"line_id": ["S1", "S41"]})
    stations = pd.DataFrame({"station_id": ["S1", "S2", "S3"]})
    orphans = trip_reference_orphans(trips, lines, stations)
    assert all(count == 0 for count, _missing in orphans.values())


def test_merge_weather_uses_the_floored_hour():
    trips = pd.DataFrame(
        {
            "trip_id": [1],
            "scheduled_departure_time": pd.to_datetime(["2024-01-01 14:37:00"]),
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 14:00:00", "2024-01-01 15:00:00"]),
            "temperature_c": [8.0, 9.0],
        }
    )
    merged = merge_weather(trips, weather)
    assert merged.loc[0, "temperature_c"] == 8.0
    assert len(merged) == 1


def test_flag_incidents_is_line_scoped_and_half_open():
    trips = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "line_id": ["S1", "S2", "S1", "S1"],
            "scheduled_departure_time": pd.to_datetime(
                [
                    "2024-01-13 09:40:00",
                    "2024-01-13 09:40:00",
                    "2024-01-13 09:23:00",
                    "2024-01-13 10:23:00",
                ]
            ),
        }
    )
    incidents = pd.DataFrame(
        {
            "incident_id": ["i1"],
            "line_id": ["S1"],
            "timestamp": pd.to_datetime(["2024-01-13 09:23:00"]),
            "incident_type": ["Signal Failure"],
            "delay_impact_factor": [1.5],
        }
    )
    flagged = flag_incidents(trips, incidents, window_minutes=60)
    assert "delay_impact_factor" not in flagged.columns
    by_id = flagged.set_index("trip_id")
    assert by_id.loc[1, "has_incident"] == 1
    assert by_id.loc[1, "incident_type"] == "Signal Failure"
    assert by_id.loc[2, "has_incident"] == 0
    assert pd.isna(by_id.loc[2, "incident_type"])
    assert by_id.loc[3, "has_incident"] == 1
    assert by_id.loc[4, "has_incident"] == 0


def test_flag_events_matches_line_membership():
    trips = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4, 5],
            "line_id": ["S1", "S2", "S41", "S5", "S1"],
            "scheduled_departure_time": pd.to_datetime(
                [
                    "2024-09-06 08:00:00",
                    "2024-09-06 08:00:00",
                    "2024-09-06 08:00:00",
                    "2024-09-06 08:00:00",
                    "2024-09-07 08:00:00",
                ]
            ),
        }
    )
    events = pd.DataFrame(
        {
            "event_name": ["IFA Berlin"],
            "date": pd.to_datetime(["2024-09-06"]),
            "impact_on_lines": ["S1,S2,S41"],
        }
    )
    flagged = flag_events(trips, events).set_index("trip_id")
    assert flagged.loc[1, "is_event_day"] == 1
    assert flagged.loc[2, "is_event_day"] == 1
    assert flagged.loc[3, "is_event_day"] == 1
    assert flagged.loc[[1, 2, 3], "event_name"].eq("IFA Berlin").all()
    assert flagged.loc[4, "is_event_day"] == 0
    assert pd.isna(flagged.loc[4, "event_name"])
    assert flagged.loc[5, "is_event_day"] == 0


def test_flag_strikes_uses_a_half_open_window():
    trips = pd.DataFrame(
        {
            "trip_id": [1, 2, 3],
            "scheduled_departure_time": pd.to_datetime(
                [
                    "2024-03-15 00:00:00",
                    "2024-03-16 12:00:00",
                    "2024-03-17 00:00:00",
                ]
            ),
        }
    )
    strikes = pd.DataFrame(
        {
            "strike_start": pd.to_datetime(["2024-03-15 00:00:00"]),
            "strike_end": pd.to_datetime(["2024-03-17 00:00:00"]),
        }
    )
    flagged = flag_strikes(trips, strikes).set_index("trip_id")
    assert flagged.loc[1, "is_strike_day"] == 1
    assert flagged.loc[2, "is_strike_day"] == 1
    assert flagged.loc[3, "is_strike_day"] == 0


def test_join_sequence_keeps_one_row_per_trip():
    trips = pd.DataFrame(
        {
            "trip_id": [1, 2],
            "line_id": ["S1", "S5"],
            "start_station_id": ["S1", "S2"],
            "end_station_id": ["S2", "S1"],
            "scheduled_departure_time": pd.to_datetime(
                ["2024-09-06 14:37:00", "2024-03-15 00:00:00"]
            ),
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-09-06 14:00:00", "2024-03-15 00:00:00"]),
            "temperature_c": [18.0, 4.0],
        }
    )
    incidents = pd.DataFrame(
        {
            "incident_id": ["i1"],
            "line_id": ["S1"],
            "timestamp": pd.to_datetime(["2024-09-06 14:00:00"]),
            "incident_type": ["Technical Fault"],
            "delay_impact_factor": [2.0],
        }
    )
    events = pd.DataFrame(
        {
            "event_name": ["IFA Berlin"],
            "date": pd.to_datetime(["2024-09-06"]),
            "impact_on_lines": ["S1, S2, S41"],
        }
    )
    strikes = pd.DataFrame(
        {
            "strike_start": pd.to_datetime(["2024-03-15 00:00:00"]),
            "strike_end": pd.to_datetime(["2024-03-17 00:00:00"]),
        }
    )
    lines = pd.DataFrame(
        {
            "line_id": ["S1", "S5"],
            "line_name": ["S1", "S5"],
            "is_ring_line": [0, 0],
            "delay_propensity": [0.7, 0.8],
        }
    )
    stations = pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "name": ["Alpha", "Beta"],
            "is_major_hub": [1, 0],
            "location_category": ["central", "suburb"],
        }
    )
    merged = merge_weather(trips, weather)
    merged = flag_incidents(merged, incidents)
    merged = flag_events(merged, events)
    merged = flag_strikes(merged, strikes)
    merged = merge_metadata(merged, stations, lines)
    assert len(merged) == len(trips)
    assert merged["trip_id"].is_unique
    assert "delay_propensity" not in merged.columns
    assert "delay_impact_factor" not in merged.columns
