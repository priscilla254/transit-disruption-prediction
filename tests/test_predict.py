import pandas as pd
import pytest

from sbahn.predict import score_batch

PARAMS = {
    "n_estimators": 8,
    "learning_rate": 0.1,
    "num_leaves": 4,
    "min_child_samples": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "scale_pos_weight": 1.0,
}
CUTOFF = "2024-11-01"


def _frame() -> pd.DataFrame:
    moments = list(pd.date_range("2024-01-01", periods=20, freq="7D"))
    rows = []
    for index, moment in enumerate(moments):
        rows.append(_row(index, moment))
    rows.append(_row(100, pd.Timestamp("2024-12-01")))
    rows.append(_row(101, pd.Timestamp("2024-11-01")))
    return pd.DataFrame(rows)


def _row(trip_id: int, moment: pd.Timestamp) -> dict:
    return {
        "trip_id": trip_id,
        "scheduled_departure_time": moment,
        "is_delayed": int(trip_id % 3 == 0),
        "line_id": "S1" if trip_id % 2 == 0 else "S41",
        "hour": int(trip_id % 24),
        "is_strike_day": int(trip_id % 5 == 0),
        "precipitation_mm": float(trip_id % 4),
        "network_load": 3 + (trip_id % 2),
    }


def test_cutoff_row_is_scored_and_earlier_rows_are_not():
    scored = score_batch(_frame(), PARAMS, CUTOFF, 0.30)
    assert scored["trip_id"].tolist() == [100, 101]
    assert list(scored.columns) == ["trip_id", "probability", "predicted"]


def test_predicted_follows_the_operating_threshold():
    frame = _frame()
    scored = score_batch(frame, PARAMS, CUTOFF, 0.30)
    expected = (scored["probability"] >= 0.30).astype(int)
    assert scored["predicted"].tolist() == expected.tolist()
    everyone = score_batch(frame, PARAMS, CUTOFF, 0.0)
    no_one = score_batch(frame, PARAMS, CUTOFF, 1.0)
    assert everyone["predicted"].tolist() == [1, 1]
    assert no_one["predicted"].tolist() == [0, 0]


def test_empty_side_of_the_cutoff_is_refused():
    frame = _frame()
    training = frame.loc[frame["scheduled_departure_time"] < CUTOFF]
    later = frame.loc[frame["scheduled_departure_time"] >= CUTOFF]
    with pytest.raises(ValueError, match="scored set is empty"):
        score_batch(training, PARAMS, CUTOFF, 0.30)
    with pytest.raises(ValueError, match="training set is empty"):
        score_batch(later, PARAMS, CUTOFF, 0.30)
