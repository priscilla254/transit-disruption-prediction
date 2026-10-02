import pandas as pd

from sbahn.features.groups import FEATURE_COLUMNS, LEAKAGE_COLUMNS
from sbahn.models.tune import model_feature_columns


def test_leakage_columns_are_not_features():
    assert LEAKAGE_COLUMNS.isdisjoint(FEATURE_COLUMNS)


def test_delay_outcomes_and_proxies_stay_out():
    blocked = {
        "is_delayed",
        "delay_minutes",
        "is_cancelled",
        "trip_id",
        "delay_impact_factor",
        "delay_propensity",
    }
    assert blocked <= LEAKAGE_COLUMNS
    assert blocked.isdisjoint(FEATURE_COLUMNS)


def test_model_inputs_drop_labels_and_name_columns():
    columns = pd.Index(
        [
            "is_delayed",
            "delay_minutes",
            "delay_propensity",
            "delay_impact_factor",
            "trip_id",
            "is_cancelled",
            "scheduled_departure_time",
            "line_name",
            "start_name",
            "end_name",
            "hour",
            "is_strike_day",
            "precipitation_mm",
        ]
    )
    usable = model_feature_columns(columns)
    assert usable == ["hour", "is_strike_day", "precipitation_mm"]
    assert set(usable).isdisjoint(LEAKAGE_COLUMNS)
