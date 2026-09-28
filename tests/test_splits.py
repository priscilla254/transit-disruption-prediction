import pandas as pd

from sbahn.models.splits import temporal_split


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
