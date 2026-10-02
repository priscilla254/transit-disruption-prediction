"""Score the chosen model once on November–December.

The classifier is the unweighted parameter set. A minute model uses those same
tree settings, without class weight, and is fit only on January–October.
`delay_minutes` is the regression target, joined from the merged trip table.
It is not a classifier input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
    root_mean_squared_error,
)

from sbahn.models.splits import temporal_split
from sbahn.models.tune import (
    CATEGORICAL_COLUMNS,
    SEED,
    _cast_categoricals,
    _fit_fold,
    choose_threshold,
    model_feature_columns,
    out_of_fold_probabilities,
    sweep_pooled_thresholds,
    write_best_params,
)

BASELINE_PARAMS = Path("reports/optuna/baseline_params.json")
MERGED_TRIPS = Path("data/interim/trips_merged.parquet")
OUT_PATH = Path("reports/evaluation/test_metrics.json")
OPERATING_PATH = Path("reports/evaluation/operating_threshold.json")
FIGURES = Path("reports/figures")


def classification_scores(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    recalls = recall_score(labels, predicted, labels=[0, 1], average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "recall_on_time": float(recalls[0]),
        "recall_delayed": float(recalls[1]),
        "delayed_f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def regression_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
    }


def attach_minutes(frame: pd.DataFrame, merged_path: Path) -> pd.DataFrame:
    minutes = pd.read_parquet(merged_path, columns=["trip_id", "delay_minutes"])
    if not minutes["trip_id"].is_unique:
        raise AssertionError("merged trip_id is not unique")
    out = frame.merge(minutes, on="trip_id", how="left", validate="one_to_one")
    if out["delay_minutes"].isna().any():
        raise AssertionError("a feature row has no delay_minutes on the merged table")
    return out


def _tree_params(params: dict[str, float | int]) -> dict[str, float | int]:
    return {key: value for key, value in params.items() if key != "scale_pos_weight"}


def _fit_regressor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict[str, float | int],
) -> tuple[LGBMRegressor, pd.DataFrame]:
    features = model_feature_columns(train.columns)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    train_x, test_x = _cast_categoricals(train[features], test[features])
    model = LGBMRegressor(
        **params,
        objective="regression",
        bagging_freq=1,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(train_x, train["delay_minutes"].astype(float), categorical_feature=categorical)
    return model, test_x


def score_holdout(
    development: pd.DataFrame,
    test: pd.DataFrame,
    params: dict[str, float | int],
    threshold: float = 0.5,
) -> dict[str, float]:
    if development["scheduled_departure_time"].max() >= test["scheduled_departure_time"].min():
        raise AssertionError("test departures are not after the development departures")
    model, test_x = _fit_fold(development, test, params)
    probabilities = model.predict_proba(test_x)[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    labels = test["is_delayed"].astype(int).to_numpy()
    scores = classification_scores(labels, predicted)
    regressor, reg_x = _fit_regressor(development, test, _tree_params(params))
    minutes = regressor.predict(reg_x)
    scores.update(regression_scores(test["delay_minutes"].to_numpy(dtype=float), minutes))
    return scores


def evaluate(
    features: pd.DataFrame,
    merged_path: Path,
    params: dict[str, float | int],
    cutoff: str,
    threshold: float,
) -> dict:
    frame = attach_minutes(features, merged_path)
    development, test = temporal_split(frame, "scheduled_departure_time", cutoff)
    scores = score_holdout(development, test, params, threshold=threshold)
    return {
        "cutoff": cutoff,
        "threshold": threshold,
        "development_rows": len(development),
        "test_rows": len(test),
        **scores,
    }


def select_operating_threshold(
    development: pd.DataFrame,
    params: dict[str, float | int],
    n_splits: int,
    cutoff: str,
) -> dict:
    """Pick the delayed-class F1 cutoff on January–October out-of-fold scores."""
    labels, probabilities, times = out_of_fold_probabilities(development, params, n_splits)
    if pd.Timestamp(times.max()) >= pd.Timestamp(cutoff):
        raise AssertionError("threshold choice included a row on or after the cutoff")
    scored = sweep_pooled_thresholds(labels, probabilities)
    chosen = choose_threshold(scored)
    at_half = next(row["f1"] for row in scored if row["threshold"] == 0.5)
    return {
        "threshold": chosen["threshold"],
        "development_f1": chosen["f1"],
        "development_f1_at_0_5": at_half,
        "thresholds": scored,
    }


def save_confusion_matrix(
    labels: np.ndarray, predicted: np.ndarray, path: Path, threshold: float
) -> np.ndarray:
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["On time", "Delayed"])
    axis.set_yticks([0, 1], ["On time", "Delayed"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"November–December at threshold {threshold:.2f}")
    darkest = float(matrix.max())
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > darkest / 2 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color=color,
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return matrix


def save_calibration_curve(
    labels: np.ndarray, probabilities: np.ndarray, path: Path, n_bins: int = 10
) -> dict[str, list[float]]:
    import matplotlib.pyplot as plt

    fraction_positive, mean_predicted = calibration_curve(
        labels, probabilities, n_bins=n_bins, strategy="quantile"
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Matches the share delayed")
    axis.plot(mean_predicted, fraction_positive, marker="o", label="Unweighted model")
    axis.set_xlabel("Mean predicted probability of delay")
    axis.set_ylabel("Share of trips that were delayed")
    axis.set_title("Calibration on November–December")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "mean_predicted": [float(value) for value in mean_predicted],
        "fraction_delayed": [float(value) for value in fraction_positive],
    }


def operating_report(
    features: pd.DataFrame,
    params: dict[str, float | int],
    cutoff: str,
    n_splits: int,
    figure_dir: Path = FIGURES,
) -> dict:
    development, test = temporal_split(features, "scheduled_departure_time", cutoff)
    if test["scheduled_departure_time"].min() < pd.Timestamp(cutoff):
        raise AssertionError("holdout includes a row before the cutoff")
    selection = select_operating_threshold(development, params, n_splits, cutoff)
    model, test_x = _fit_fold(development, test, params)
    probabilities = model.predict_proba(test_x)[:, 1]
    labels = test["is_delayed"].astype(int).to_numpy()
    predicted = (probabilities >= selection["threshold"]).astype(int)
    matrix = save_confusion_matrix(
        labels, predicted, figure_dir / "confusion_matrix.png", selection["threshold"]
    )
    calibration = save_calibration_curve(
        labels, probabilities, figure_dir / "calibration_curve.png"
    )
    scores = classification_scores(labels, predicted)
    return {
        "cutoff": cutoff,
        "n_splits": n_splits,
        "threshold": selection["threshold"],
        "development_f1": selection["development_f1"],
        "development_f1_at_0_5": selection["development_f1_at_0_5"],
        "thresholds": selection["thresholds"],
        "test_rows": len(test),
        "test_macro_f1": scores["macro_f1"],
        "test_recall_on_time": scores["recall_on_time"],
        "test_recall_delayed": scores["recall_delayed"],
        "test_delayed_f1": scores["delayed_f1"],
        "confusion_matrix": {
            "labels": ["on_time", "delayed"],
            "counts": matrix.astype(int).tolist(),
        },
        "calibration": calibration,
    }


def log_final_metrics(payload: dict) -> None:
    import mlflow

    mlflow.log_params(
        {
            "cutoff": payload["cutoff"],
            "threshold": payload["threshold"],
            "model": "no-weighting",
        }
    )
    for name in ("macro_f1", "recall_on_time", "recall_delayed", "delayed_f1", "mae", "rmse"):
        mlflow.log_metric(name, float(payload[name]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the chosen model once on the holdout.")
    parser.add_argument("--in", dest="in_path", default="data/processed/trips_features.parquet")
    parser.add_argument("--merged", default=str(MERGED_TRIPS))
    parser.add_argument("--params", default=str(BASELINE_PARAMS))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    saved = json.loads(Path(args.params).read_text(encoding="utf-8"))
    features = pd.read_parquet(args.in_path)
    payload = evaluate(
        features,
        Path(args.merged),
        saved["best_params"],
        cutoff=saved["cutoff"],
        threshold=float(saved.get("threshold", 0.5)),
    )
    write_best_params(Path(args.out), payload)
    print(
        f"macro-F1 {payload['macro_f1']:.4f}  "
        f"recall on-time {payload['recall_on_time']:.4f}  "
        f"recall delayed {payload['recall_delayed']:.4f}"
    )
    print(f"MAE {payload['mae']:.4f}  RMSE {payload['rmse']:.4f}")
    report = operating_report(
        features,
        saved["best_params"],
        cutoff=saved["cutoff"],
        n_splits=int(saved["n_splits"]),
    )
    write_best_params(OPERATING_PATH, report)
    print(
        f"operating threshold {report['threshold']:.2f}  "
        f"development F1 {report['development_f1']:.4f}  "
        f"test delayed F1 {report['test_delayed_f1']:.4f}"
    )
    from sbahn.models.tracking import study_run

    with study_run("final-test", {"treatment": "final-test", "source": "live"}):
        log_final_metrics(payload)


if __name__ == "__main__":
    main()
