"""Pandera checks for the cleaned trips table, run before any join."""

import pandera.pandas as pa

trips_schema = pa.DataFrameSchema(
    {
        "trip_id": pa.Column(int, unique=True, nullable=False),
        "line_id": pa.Column(str, nullable=False),
        "scheduled_departure_time": pa.Column(pa.DateTime, nullable=False),
        "delay_minutes": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "is_delayed": pa.Column(int, nullable=False, checks=pa.Check.isin([0, 1])),
    },
    strict=False,
)
