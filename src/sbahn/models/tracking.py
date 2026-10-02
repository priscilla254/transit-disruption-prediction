"""Record tuning runs in a local MLflow store.

The store is a SQLite file at `mlruns/mlflow.db`. Each treatment is one parent
run. Each Optuna trial inside a live search is a nested run. Reading the JSON
already written under `reports/optuna/` records the finished studies without
starting another search.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import optuna

EXPERIMENT_NAME = "sbahn-delay"
TRACKING_URI = "sqlite:///mlruns/mlflow.db"
REPORTS_DIR = Path("reports/optuna")

SAVED_SEARCHES = (
    ("class-weights", "best_params.json"),
    ("no-weighting", "baseline_params.json"),
    ("weights-plus-resampling", "resample_params.json"),
)
SAVED_SWEEP = ("threshold-sweep", "threshold_sweep.json")


def use_store(uri: str | None = None, experiment: str = EXPERIMENT_NAME) -> None:
    store_uri = uri or TRACKING_URI
    if store_uri.startswith("sqlite:///"):
        Path(store_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store_uri)
    mlflow.set_experiment(experiment)


def trial_callback(scale_pos_weight: float | None, oversample: bool):
    """Log one nested run for each completed Optuna trial."""

    def _callback(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            params = {key: _mlflow_value(value) for key, value in trial.params.items()}
            if scale_pos_weight is not None and "scale_pos_weight" not in params:
                params["scale_pos_weight"] = float(scale_pos_weight)
            params["oversample"] = bool(oversample)
            mlflow.log_params(params)
            mlflow.set_tag("trial_number", str(trial.number))
            if trial.value is not None:
                mlflow.log_metric("mean_f1", float(trial.value))

    return _callback


@contextmanager
def study_run(
    name: str,
    tags: dict[str, str],
    uri: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    use_store(uri)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(tags)
        yield run


def log_search_result(payload: dict) -> None:
    """Log a finished Optuna search into the active run."""
    params: dict[str, str | int | float | bool] = {
        "cutoff": str(payload["cutoff"]),
        "n_splits": int(payload["n_splits"]),
        "n_trials": int(payload["n_trials"]),
        "threshold": _mlflow_value(payload.get("threshold", 0.5)),
        "oversample": bool(payload.get("oversample", False)),
    }
    for key, value in payload["best_params"].items():
        params[key] = _mlflow_value(value)
    mlflow.log_params(params)
    mlflow.log_metric("development_f1", float(payload["best_mean_f1"]))
    mlflow.log_metric("test_f1", float(payload["test_f1"]))


def log_sweep_result(payload: dict) -> None:
    """Log a threshold sweep into the active run."""
    mlflow.log_params(
        {
            "cutoff": str(payload["cutoff"]),
            "n_splits": int(payload["n_splits"]),
            "scale_pos_weight": _mlflow_value(payload["scale_pos_weight"]),
            "threshold": _mlflow_value(payload["best_threshold"]),
            "distance_from_0_5": _mlflow_value(payload["distance_from_0_5"]),
        }
    )
    mlflow.log_metric("development_f1", float(payload["development_f1"]))
    mlflow.log_metric("test_f1", float(payload["test_f1"]))
    for row in payload["thresholds"]:
        step = round(float(row["threshold"]) * 100)
        mlflow.log_metric("pooled_f1", float(row["f1"]), step=step)


def backfill(uri: str | None = None, reports_dir: Path = REPORTS_DIR) -> list[str]:
    """Record the saved JSON studies. Skip a treatment that is already stored."""
    use_store(uri)
    created: list[str] = []
    for name, filename in SAVED_SEARCHES:
        run_id = _log_saved(name, reports_dir / filename, kind="search", uri=uri)
        if run_id is not None:
            created.append(run_id)
    name, filename = SAVED_SWEEP
    run_id = _log_saved(name, reports_dir / filename, kind="sweep", uri=uri)
    if run_id is not None:
        created.append(run_id)
    return created


def _log_saved(name: str, path: Path, kind: str, uri: str | None) -> str | None:
    if _saved_exists(name):
        print(f"already logged {name}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    tags = {"treatment": name, "source": "saved-json", "saved_study": name}
    with study_run(name, tags, uri=uri) as run:
        if kind == "search":
            log_search_result(payload)
        else:
            log_sweep_result(payload)
        run_id = run.info.run_id
    print(f"logged {name} as {run_id}")
    return run_id


def _saved_exists(name: str) -> bool:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return False
    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.saved_study = '{name}'",
        max_results=1,
    )
    return not found.empty


def _mlflow_value(value: object) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def main() -> None:
    backfill()


if __name__ == "__main__":
    main()
