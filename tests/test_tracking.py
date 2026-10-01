import json

import mlflow
import pandas as pd
import pytest

from sbahn.models.splits import temporal_split
from sbahn.models.tracking import backfill, log_search_result, study_run
from sbahn.models.tune import run_study
from test_tune import _frame


def _uri(tmp_path) -> str:
    return "sqlite:///" + (tmp_path / "mlflow.db").as_posix()


def _search_payload(**overrides) -> dict:
    payload = {
        "cutoff": "2024-11-01",
        "n_splits": 5,
        "n_trials": 40,
        "best_mean_f1": 0.379,
        "best_params": {"n_estimators": 89, "learning_rate": 0.018, "scale_pos_weight": 22.2},
        "test_f1": 0.818,
    }
    payload.update(overrides)
    return payload


def test_untracked_study_does_not_start_a_run(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("mlflow run started")

    monkeypatch.setattr(mlflow, "start_run", fail)
    development, _test = temporal_split(_frame(), "scheduled_departure_time", "2024-11-01")
    run_study(development, n_splits=2, n_trials=1, seed=0)


def test_saved_studies_log_once(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for filename in ("best_params.json", "baseline_params.json", "resample_params.json"):
        (reports / filename).write_text(json.dumps(_search_payload()), encoding="utf-8")
    sweep = {
        "cutoff": "2024-11-01",
        "n_splits": 5,
        "scale_pos_weight": 22.2,
        "thresholds": [
            {"threshold": 0.5, "f1": 0.634},
            {"threshold": 0.55, "f1": 0.587},
        ],
        "best_threshold": 0.5,
        "distance_from_0_5": 0.0,
        "development_f1": 0.634,
        "test_f1": 0.818,
    }
    (reports / "threshold_sweep.json").write_text(json.dumps(sweep), encoding="utf-8")

    uri = _uri(tmp_path)
    created = backfill(uri=uri, reports_dir=reports)
    assert len(created) == 4
    assert backfill(uri=uri, reports_dir=reports) == []

    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name("sbahn-delay")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 4
    assert set(runs["tags.treatment"]) == {
        "class-weights",
        "no-weighting",
        "weights-plus-resampling",
        "threshold-sweep",
    }
    weighted = runs[runs["tags.treatment"] == "class-weights"].iloc[0]
    assert weighted["metrics.development_f1"] == pytest.approx(0.379)
    assert weighted["metrics.test_f1"] == pytest.approx(0.818)
    assert float(weighted["params.threshold"]) == pytest.approx(0.5)
    assert float(weighted["params.scale_pos_weight"]) == pytest.approx(22.2)

    sweep_id = runs[runs["tags.treatment"] == "threshold-sweep"].iloc[0]["run_id"]
    history = {metric.step: metric.value for metric in mlflow.MlflowClient().get_metric_history(sweep_id, "pooled_f1")}
    assert history[50] == pytest.approx(0.634)
    assert history[55] == pytest.approx(0.587)


def test_live_study_logs_a_run_per_trial(tmp_path):
    development, _test = temporal_split(_frame(), "scheduled_departure_time", "2024-11-01")
    uri = _uri(tmp_path)
    with study_run("no-weighting", {"treatment": "no-weighting", "source": "live"}, uri=uri):
        result = run_study(
            development,
            n_splits=2,
            n_trials=2,
            seed=0,
            scale_pos_weight=1.0,
            track=True,
        )
        log_search_result(
            {
                "cutoff": "2024-11-01",
                "n_splits": 2,
                "n_trials": 2,
                "oversample": False,
                "best_mean_f1": result.best_mean_f1,
                "best_params": result.best_params,
                "test_f1": 0.25,
            }
        )

    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name("sbahn-delay")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    children = runs[runs["tags.mlflow.parentRunId"].notna()]
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert len(children) == 2
    assert children["metrics.mean_f1"].notna().all()
    assert children["params.scale_pos_weight"].astype(float).eq(1.0).all()
    assert parent["metrics.test_f1"] == pytest.approx(0.25)
    assert parent["metrics.development_f1"] == pytest.approx(result.best_mean_f1)
    assert pd.notna(parent["params.n_estimators"])
