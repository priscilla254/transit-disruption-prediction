import pandas as pd
import pytest

from sbahn.models.slices import disruption_label, group_metrics, weather_band


def test_weather_band_edges():
    precipitation = pd.Series([0.0, 0.0, 1.0, 2.0, 2.01])
    assert weather_band(precipitation).tolist() == ["none", "none", "light", "light", "heavy"]


def test_strike_outranks_another_flag():
    frame = pd.DataFrame(
        {
            "is_strike_day": [1, 0, 0],
            "has_incident": [1, 1, 0],
            "is_event_day": [1, 0, 1],
        }
    )
    assert disruption_label(frame).tolist() == ["strike", "incident", "event"]


def test_group_metrics_match_a_fixed_prediction():
    frame = pd.DataFrame(
        {
            "line_id": ["S1", "S1", "S1", "S1"],
            "is_delayed": [0, 0, 1, 1],
            "predicted": [0, 1, 1, 0],
        }
    )
    rows = group_metrics(frame, "line_id", ["S1", "S2"])
    assert rows[0]["n"] == 4
    assert rows[0]["delay_rate"] == pytest.approx(0.5)
    assert rows[0]["precision"] == pytest.approx(0.5)
    assert rows[0]["recall"] == pytest.approx(0.5)
    assert rows[0]["f1"] == pytest.approx(0.5)
    assert rows[1] == {
        "group": "S2",
        "n": 0,
        "delay_rate": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
