import pandas as pd


def temporal_split(
    frame: pd.DataFrame, time_column: str, train_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put earlier departures in train and later departures in test."""
    times = pd.to_datetime(frame[time_column])
    cutoff = pd.Timestamp(train_end)
    train = frame.loc[times < cutoff].copy()
    test = frame.loc[times >= cutoff].copy()
    return train, test
