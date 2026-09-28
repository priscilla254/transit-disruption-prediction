from sbahn.features.groups import FEATURE_COLUMNS, LEAKAGE_COLUMNS


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
