import pandas as pd

from sbahn.models.splits import temporal_split
from sbahn.models.tune import iter_folds


def test_train_departures_are_before_test():
    frame = pd.DataFrame(
        {
            "scheduled_departure_time": pd.to_datetime(
                ["2024-01-01", "2024-06-01", "2024-09-01", "2024-12-01"]
            ),
            "is_delayed": [0, 1, 0, 1],
        }
    )
    train, test = temporal_split(frame, "scheduled_departure_time", "2024-07-01")
    assert len(train) == 2
    assert len(test) == 2
    assert train["scheduled_departure_time"].max() < test["scheduled_departure_time"].min()


def test_cutoff_timestamp_goes_to_the_test_set():
    frame = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "scheduled_departure_time": pd.to_datetime(
                [
                    "2024-10-31 23:59:00",
                    "2024-11-01 00:00:00",
                    "2024-11-01 00:00:00",
                    "2024-11-02 08:00:00",
                ]
            ),
        }
    )
    train, test = temporal_split(frame, "scheduled_departure_time", "2024-11-01")
    assert train["trip_id"].tolist() == [1]
    assert test["trip_id"].tolist() == [2, 3, 4]
    assert train["scheduled_departure_time"].max() < pd.Timestamp("2024-11-01")
    assert test["scheduled_departure_time"].min() == pd.Timestamp("2024-11-01")


def test_shared_fold_boundary_stays_in_training():
    times = [
        "2024-01-01 00:00:00",
        "2024-01-02 00:00:00",
        "2024-01-03 00:00:00",
        "2024-01-03 00:00:00",
        "2024-01-03 00:00:00",
        "2024-01-04 00:00:00",
        "2024-01-05 00:00:00",
        "2024-01-06 00:00:00",
    ]
    frame = pd.DataFrame(
        {
            "trip_id": list(range(1, 9)),
            "scheduled_departure_time": pd.to_datetime(times),
            "is_delayed": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    folds = list(iter_folds(frame, n_splits=3))
    _first, second, _third = folds
    train, validation = second
    assert set(train["trip_id"]) == {1, 2, 3, 4, 5}
    assert validation["trip_id"].tolist() == [6]
    for train_rows, validation_rows in folds:
        assert (
            train_rows["scheduled_departure_time"].max()
            < validation_rows["scheduled_departure_time"].min()
        )
