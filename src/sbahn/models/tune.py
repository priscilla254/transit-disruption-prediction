"""Search LightGBM hyperparameters on a time-ordered development set.

January through October is the development set. Optuna maximizes mean F1 across
TimeSeriesSplit folds. November and December are scored once, after the search.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from sbahn.features.groups import FEATURE_COLUMNS
from sbahn.models.splits import temporal_split

SEED = 42
EXCLUDED_FEATURES = frozenset(
    {"scheduled_departure_time", "line_name", "start_name", "end_name"}
)
CATEGORICAL_COLUMNS = (
    "line_id",
    "start_station_id",
    "end_station_id",
    "route_key",
    "location_pair",
    "hub_pattern",
    "start_location_category",
    "end_location_category",
    "weather_condition",
    "incident_type",
    "event_name",
)


@dataclass(frozen=True)
class StudyResult:
    best_params: dict[str, float | int]
    best_mean_f1: float


def model_feature_columns(columns: pd.Index) -> list[str]:
    usable = FEATURE_COLUMNS - EXCLUDED_FEATURES
    return sorted(column for column in columns if column in usable)


def iter_folds(frame: pd.DataFrame, n_splits: int):
    """Yield training and validation rows in time order.

    Rows that share the boundary timestamp stay in training, so every
    validation departure is strictly later.
    """
    ordered = frame.sort_values("scheduled_departure_time", kind="mergesort")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_pos, val_pos) in enumerate(splitter.split(ordered), start=1):
        times = ordered["scheduled_departure_time"]
        boundary = times.iloc[train_pos].max()
        val_times = times.iloc[val_pos]
        later = (val_times > boundary).to_numpy()
        train_pos = np.concatenate([train_pos, val_pos[~later]])
        val_pos = val_pos[later]
        if len(val_pos) == 0:
            raise AssertionError(f"fold {fold} validation set is empty after the time cut")
        train = ordered.iloc[train_pos]
        validation = ordered.iloc[val_pos]
        if train["scheduled_departure_time"].max() >= validation["scheduled_departure_time"].min():
            raise AssertionError(
                f"fold {fold} validation departure is not after the training departures"
            )
        yield train, validation


def suggest_params(
    trial: optuna.Trial, scale_pos_weight: float | None = None
) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if scale_pos_weight is None:
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 40.0)
    else:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def _cast_categoricals(
    train: pd.DataFrame, other: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    other = other.copy()
    for column in CATEGORICAL_COLUMNS:
        if column not in train.columns:
            continue
        categories = pd.Index(train[column].dropna().unique())
        known = set(categories)
        dtype = pd.CategoricalDtype(categories)
        train[column] = train[column].where(train[column].isin(known)).astype(dtype)
        other[column] = other[column].where(other[column].isin(known)).astype(dtype)
    return train, other


def _classifier(params: dict[str, float | int]) -> LGBMClassifier:
    return LGBMClassifier(
        **params,
        objective="binary",
        bagging_freq=1,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )


def oversample_delayed(train: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Copy delayed training rows until they match the on-time count."""
    labels = train["is_delayed"].astype(int)
    delayed = train.loc[labels == 1]
    on_time_count = int((labels == 0).sum())
    if delayed.empty or len(delayed) >= on_time_count:
        return train
    extra = delayed.sample(n=on_time_count - len(delayed), replace=True, random_state=seed)
    return pd.concat([train, extra], ignore_index=True)


def _fit_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict[str, float | int],
    oversample: bool = False,
    seed: int = SEED,
) -> tuple[LGBMClassifier, pd.DataFrame]:
    features = model_feature_columns(train.columns)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    train_rows = oversample_delayed(train, seed) if oversample else train
    train_x, val_x = _cast_categoricals(train_rows[features], validation[features])
    model = _classifier(params)
    model.fit(train_x, train_rows["is_delayed"].astype(int), categorical_feature=categorical)
    return model, val_x


def fold_f1(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict[str, float | int],
    oversample: bool = False,
    seed: int = SEED,
) -> float:
    model, val_x = _fit_fold(train, validation, params, oversample=oversample, seed=seed)
    predicted = model.predict(val_x)
    return float(f1_score(validation["is_delayed"].astype(int), predicted, zero_division=0))


def pooled_f1(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    """Score F1 on the concatenated out-of-fold predictions, not the mean of the fold F1s."""
    predicted = (probabilities >= threshold).astype(int)
    return float(f1_score(labels, predicted, zero_division=0))


def out_of_fold_probabilities(
    development: pd.DataFrame, params: dict[str, float | int], n_splits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for fold, (train, validation) in enumerate(iter_folds(development, n_splits), start=1):
        model, val_x = _fit_fold(train, validation, params, seed=SEED + fold)
        labels.append(validation["is_delayed"].astype(int).to_numpy())
        probabilities.append(model.predict_proba(val_x)[:, 1])
        times.append(validation["scheduled_departure_time"].to_numpy())
    return np.concatenate(labels), np.concatenate(probabilities), np.concatenate(times)


def sweep_pooled_thresholds(
    labels: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, float]]:
    thresholds = [round(0.1 + 0.05 * step, 2) for step in range(17)]
    return [
        {"threshold": threshold, "f1": pooled_f1(labels, probabilities, threshold)}
        for threshold in thresholds
    ]


def choose_threshold(scored: list[dict[str, float]]) -> dict[str, float]:
    best_f1 = max(row["f1"] for row in scored)
    tied = [row for row in scored if row["f1"] == best_f1]
    return min(tied, key=lambda row: (abs(row["threshold"] - 0.5), row["threshold"]))


def mean_f1(
    frame: pd.DataFrame,
    params: dict[str, float | int],
    n_splits: int,
    oversample: bool = False,
) -> float:
    scores = [
        fold_f1(train, validation, params, oversample=oversample, seed=SEED + fold)
        for fold, (train, validation) in enumerate(iter_folds(frame, n_splits), start=1)
    ]
    return float(np.mean(scores))


def run_study(
    frame: pd.DataFrame,
    n_splits: int,
    n_trials: int,
    seed: int = SEED,
    scale_pos_weight: float | None = None,
    oversample: bool = False,
    track: bool = False,
) -> StudyResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, scale_pos_weight)
        return mean_f1(frame, params, n_splits, oversample=oversample)

    def report(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"trial {trial.number}: mean F1 {trial.value:.4f}")

    callbacks = [report]
    if track:
        from sbahn.models.tracking import trial_callback

        callbacks.append(trial_callback(scale_pos_weight, oversample))
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
    params = dict(study.best_params)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return StudyResult(best_params=params, best_mean_f1=float(study.best_value))


def fit_and_score(
    development: pd.DataFrame,
    test: pd.DataFrame,
    params: dict[str, float | int],
    oversample: bool = False,
    threshold: float = 0.5,
) -> float:
    model, test_x = _fit_fold(development, test, params, oversample=oversample)
    probabilities = model.predict_proba(test_x)[:, 1]
    return pooled_f1(test["is_delayed"].astype(int).to_numpy(), probabilities, threshold)


def _plain_params(params: dict[str, float | int]) -> dict[str, float | int]:
    plain: dict[str, float | int] = {}
    for name, value in params.items():
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            plain[name] = int(value)
        else:
            plain[name] = float(value)
    return plain


def write_best_params(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tune(
    frame: pd.DataFrame, cutoff: str, n_splits: int, n_trials: int, track: bool = False
) -> dict:
    development, test = temporal_split(frame, "scheduled_departure_time", cutoff)
    result = run_study(development, n_splits=n_splits, n_trials=n_trials, track=track)
    test_f1 = fit_and_score(development, test, result.best_params)
    print(f"best mean F1 {result.best_mean_f1:.4f}")
    print("best params: " + json.dumps(_plain_params(result.best_params), sort_keys=True))
    print(f"test F1 {test_f1:.4f}")
    return {
        "cutoff": cutoff,
        "n_splits": n_splits,
        "n_trials": n_trials,
        "best_mean_f1": result.best_mean_f1,
        "best_params": _plain_params(result.best_params),
        "test_f1": test_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LightGBM with time-series folds.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--cutoff", default="2024-11-01")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--out", default="reports/optuna/best_params.json")
    args = parser.parse_args()
    frame = pd.read_parquet(args.in_path)
    from sbahn.models.tracking import log_search_result, study_run

    with study_run("class-weights", {"treatment": "class-weights", "source": "live"}):
        payload = tune(frame, args.cutoff, args.n_splits, args.n_trials, track=True)
        write_best_params(Path(args.out), payload)
        log_search_result(payload)


if __name__ == "__main__":
    main()
