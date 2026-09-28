"""Columns allowed as model inputs.

The rule is in docs/feature_dictionary.md: a value is usable only if it would
already be known on the platform before scheduled departure, and it must not
stand in for the delay label.
"""

LEAKAGE_COLUMNS = frozenset(
    {
        "is_delayed",
        "delay_minutes",
        "is_cancelled",
        "trip_id",
        "delay_impact_factor",
        "delay_propensity",
    }
)

# Timetable and lookups. Time-gated fields (weather, incidents, events, strikes)
# are joined later and are not listed as raw trip columns.
FEATURE_COLUMNS = frozenset(
    {
        "line_id",
        "start_station_id",
        "end_station_id",
        "scheduled_departure_time",
        "is_peak_hour",
        "line_name",
        "is_ring_line",
        "start_name",
        "end_name",
        "start_is_major_hub",
        "end_is_major_hub",
        "start_location_category",
        "end_location_category",
        "temperature_c",
        "precipitation_mm",
        "wind_speed_kmh",
        "weather_condition",
        "has_incident",
        "incident_type",
        "is_event_day",
        "event_name",
        "is_strike_day",
    }
)
