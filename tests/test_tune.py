import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from sbahn.models.splits import temporal_split
from sbahn.models.compare_imbalance import run_threshold_sweep
from sbahn.models.tune import (
    iter_folds,
    out_of_fold_probabilities,
    oversample_delayed,
    pooled_f1,
    run_study,
)


def _frame() -> pd.DataFrame:
    development = pd.date_range("2024-01-01", "2024-10-21", freq="10D")
    times = list(development) + [pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-15")]
    rows = []
    for trip_id, moment in enumerate(times, start=1):
        rows.append(
            {
                "trip_id": trip_id,
                "scheduled_departure_time": moment,
                "is_delayed": trip_id % 2,
                "line_id": "S1" if trip_id % 2 else "S2",
                "hour": moment.hour,
                "is_peak_hour": 0,
                "precipitation_mm": 0.0,
                "network_load": 4,
            }
        )
    return pd.DataFrame(rows)


def test_december_rows_stay_out_of_the_search():
    frame = _frame()
    development, test = temporal_split(frame, "scheduled_departure_time", "2024-11-01")
    assert test["scheduled_departure_time"].min() >= pd.Timestamp("2024-11-01")
    december_ids = set(frame.loc[frame["scheduled_departure_time"] >= "2024-12-01", "trip_id"])
    assert december_ids
    for train, validation in iter_folds(development, n_splits=2):
        assert train["scheduled_departure_time"].max() < validation["scheduled_departure_time"].min()
        seen = set(train["trip_id"]).union(validation["trip_id"])
        assert december_ids.isdisjoint(seen)


def test_study_returns_a_finite_mean_f1():
    development, _test = temporal_split(_frame(), "scheduled_departure_time", "2024-11-01")
    result = run_study(development, n_splits=2, n_trials=2, seed=0)
    assert math.isfinite(result.best_mean_f1)
    assert "n_estimators" in result.best_params
    assert "scale_pos_weight" in result.best_params


def test_oversample_balances_training_and_leaves_validation():
    development, _test = temporal_split(_frame(), "scheduled_departure_time", "2024-11-01")
    train, validation = next(iter_folds(development, n_splits=2))
    train = train.copy()
    train["is_delayed"] = 0
    train.iloc[:2, train.columns.get_loc("is_delayed")] = 1
    before = validation.copy()
    balanced = oversample_delayed(train)
    delayed = int((balanced["is_delayed"] == 1).sum())
    on_time = int((balanced["is_delayed"] == 0).sum())
    assert delayed == on_time
    assert len(balanced) > len(train)
    pd.testing.assert_frame_equal(validation, before)


def test_pooled_f1_is_not_the_mean_of_fold_f1s():
    folds = [
        (np.array([1]), np.array([0.9])),
        (np.array([1, 0, 0]), np.array([0.9, 0.9, 0.1])),
    ]
    fold_scores = [
        f1_score(labels, (probabilities >= 0.5).astype(int), zero_division=0)
        for labels, probabilities in folds
    ]
    labels = np.concatenate([labels for labels, _probabilities in folds])
    probabilities = np.concatenate([probabilities for _labels, probabilities in folds])
    assert pooled_f1(labels, probabilities, 0.5) != float(np.mean(fold_scores))


def test_out_of_fold_pool_stops_before_november():
    frame = _frame()
    development, test = temporal_split(frame, "scheduled_departure_time", "2024-11-01")
    params = {
        "n_estimators": 5,
        "learning_rate": 0.1,
        "num_leaves": 4,
        "min_child_samples": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": 1.0,
    }
    sweep = run_threshold_sweep(
        frame, "2024-11-01", n_splits=2, weighted={"best_params": params}
    )
    _labels, _probabilities, times = out_of_fold_probabilities(development, params, n_splits=2)
    assert len(test) > 0
    assert pd.Timestamp(times.max()) < pd.Timestamp("2024-11-01")
    assert sweep["best_threshold"] <= 0.9
