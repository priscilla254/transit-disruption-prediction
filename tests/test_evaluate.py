import math

import numpy as np
import pandas as pd
import pytest
from test_tune import _frame

from sbahn.models.evaluate import (
    attach_minutes,
    classification_scores,
    operating_report,
    regression_scores,
    save_calibration_curve,
    save_confusion_matrix,
    score_holdout,
    select_operating_threshold,
)
from sbahn.models.splits import temporal_split
from sbahn.models.tune import choose_threshold


def test_macro_f1_and_recalls_on_a_known_prediction():
    labels = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 0])
    scores = classification_scores(labels, predicted)
    assert scores["recall_on_time"] == 0.5
    assert scores["recall_delayed"] == 0.5
    assert scores["macro_f1"] == 0.5
    assert scores["delayed_f1"] == 0.5


def test_minute_errors_match_the_residuals():
    actual = np.array([0.0, 10.0])
    predicted = np.array([0.0, 7.0])
    scores = regression_scores(actual, predicted)
    assert scores["mae"] == 1.5
    assert scores["rmse"] == pytest.approx(1.5 * math.sqrt(2))


def test_minutes_join_keeps_one_row_per_trip(tmp_path):
    frame = _frame()[["trip_id", "scheduled_departure_time", "is_delayed"]]
    merged = tmp_path / "merged.parquet"
    minutes = frame[["trip_id"]].copy()
    minutes["delay_minutes"] = np.where(frame["is_delayed"].to_numpy() == 1, 12, 1)
    minutes.to_parquet(merged, index=False)
    attached = attach_minutes(frame, merged)
    assert len(attached) == len(frame)
    assert attached["delay_minutes"].notna().all()


def test_holdout_scores_use_only_earlier_rows():
    frame = _frame()
    frame["delay_minutes"] = np.where(frame["is_delayed"].to_numpy() == 1, 12, 1)
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
    scores = score_holdout(development, test, params)
    assert test["scheduled_departure_time"].min() >= pd.Timestamp("2024-11-01")
    assert math.isfinite(scores["macro_f1"])
    assert 0.0 <= scores["recall_on_time"] <= 1.0
    assert 0.0 <= scores["recall_delayed"] <= 1.0
    assert scores["mae"] >= 0.0
    assert scores["rmse"] + 1e-9 >= scores["mae"]


def _small_params() -> dict:
    return {
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


def test_operating_threshold_is_chosen_before_november(tmp_path):
    frame = _frame()
    development, test = temporal_split(frame, "scheduled_departure_time", "2024-11-01")
    selection = select_operating_threshold(development, _small_params(), n_splits=2, cutoff="2024-11-01")
    assert selection["threshold"] == choose_threshold(selection["thresholds"])["threshold"]
    report = operating_report(frame, _small_params(), "2024-11-01", n_splits=2, figure_dir=tmp_path)
    assert report["test_rows"] == len(test)
    assert (tmp_path / "confusion_matrix.png").is_file()
    assert (tmp_path / "calibration_curve.png").is_file()
    counts = np.array(report["confusion_matrix"]["counts"])
    assert counts.sum() == len(test)


def test_plots_use_the_given_predictions(tmp_path):
    labels = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])
    matrix = save_confusion_matrix(labels, predicted, tmp_path / "confusion_matrix.png", 0.4)
    assert matrix.tolist() == [[1, 1], [0, 2]]
    curve = save_calibration_curve(labels, np.array([0.1, 0.4, 0.6, 0.9]), tmp_path / "calibration_curve.png", n_bins=2)
    assert len(curve["mean_predicted"]) == len(curve["fraction_delayed"])
