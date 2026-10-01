"""Compare class weight, a decision threshold, and training-fold oversampling.

November and December are scored once, after each choice is fixed on the
January–October development set. The weighted Optuna file is left unchanged.
"""

import json
from pathlib import Path

import pandas as pd

from sbahn.models.splits import temporal_split
from sbahn.models.tune import (
    choose_threshold,
    fit_and_score,
    out_of_fold_probabilities,
    run_study,
    sweep_pooled_thresholds,
    write_best_params,
)

WEIGHTED_PARAMS = Path("reports/optuna/best_params.json")
BASELINE_PARAMS = Path("reports/optuna/baseline_params.json")
THRESHOLD_SWEEP = Path("reports/optuna/threshold_sweep.json")
RESAMPLE_PARAMS = Path("reports/optuna/resample_params.json")


def _load_weighted(path: Path = WEIGHTED_PARAMS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_fixed_weight_study(
    frame: pd.DataFrame,
    cutoff: str,
    n_splits: int,
    n_trials: int,
    scale_pos_weight: float,
    oversample: bool,
    track: bool = False,
) -> dict:
    development, test = temporal_split(frame, "scheduled_departure_time", cutoff)
    result = run_study(
        development,
        n_splits=n_splits,
        n_trials=n_trials,
        scale_pos_weight=scale_pos_weight,
        oversample=oversample,
        track=track,
    )
    test_f1 = fit_and_score(
        development, test, result.best_params, oversample=oversample, threshold=0.5
    )
    return {
        "cutoff": cutoff,
        "n_splits": n_splits,
        "n_trials": n_trials,
        "threshold": 0.5,
        "oversample": oversample,
        "best_mean_f1": result.best_mean_f1,
        "best_params": {
            key: (int(value) if isinstance(value, int) and not isinstance(value, bool) else float(value))
            for key, value in result.best_params.items()
        },
        "test_f1": test_f1,
    }


def run_threshold_sweep(
    frame: pd.DataFrame, cutoff: str, n_splits: int, weighted: dict | None = None
) -> dict:
    weighted = _load_weighted() if weighted is None else weighted
    development, test = temporal_split(frame, "scheduled_departure_time", cutoff)
    labels, probabilities, times = out_of_fold_probabilities(
        development, weighted["best_params"], n_splits
    )
    if pd.Timestamp(times.max()) >= pd.Timestamp(cutoff):
        raise AssertionError("threshold sweep included a row on or after the cutoff")
    scored = sweep_pooled_thresholds(labels, probabilities)
    chosen = choose_threshold(scored)
    test_f1 = fit_and_score(
        development,
        test,
        weighted["best_params"],
        threshold=chosen["threshold"],
    )
    return {
        "cutoff": cutoff,
        "n_splits": n_splits,
        "scale_pos_weight": weighted["best_params"]["scale_pos_weight"],
        "thresholds": scored,
        "best_threshold": chosen["threshold"],
        "distance_from_0_5": chosen["threshold"] - 0.5,
        "development_f1": chosen["f1"],
        "test_f1": test_f1,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare imbalance treatments.")
    parser.add_argument("--in", dest="in_path", default="data/processed/trips_features.parquet")
    parser.add_argument("--cutoff", default="2024-11-01")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=40)
    args = parser.parse_args()
    frame = pd.read_parquet(args.in_path)
    weighted = _load_weighted()
    from sbahn.models.tracking import log_search_result, log_sweep_result, study_run

    print("Baseline: scale_pos_weight fixed at 1")
    with study_run("no-weighting", {"treatment": "no-weighting", "source": "live"}):
        baseline = run_fixed_weight_study(
            frame,
            args.cutoff,
            args.n_splits,
            args.n_trials,
            scale_pos_weight=1.0,
            oversample=False,
            track=True,
        )
        write_best_params(BASELINE_PARAMS, baseline)
        log_search_result(baseline)
    print(f"baseline mean F1 {baseline['best_mean_f1']:.4f} test F1 {baseline['test_f1']:.4f}")

    print("Threshold sweep on the saved weighted parameters")
    with study_run("threshold-sweep", {"treatment": "threshold-sweep", "source": "live"}):
        sweep = run_threshold_sweep(frame, args.cutoff, args.n_splits, weighted)
        write_best_params(THRESHOLD_SWEEP, sweep)
        log_sweep_result(sweep)
    print(
        f"threshold {sweep['best_threshold']:.2f} "
        f"({sweep['distance_from_0_5']:+.2f} from 0.5) "
        f"dev F1 {sweep['development_f1']:.4f} test F1 {sweep['test_f1']:.4f}"
    )

    weight = float(weighted["best_params"]["scale_pos_weight"])
    print(f"Resampling with scale_pos_weight fixed at {weight:.4f}")
    with study_run(
        "weights-plus-resampling",
        {"treatment": "weights-plus-resampling", "source": "live"},
    ):
        resampled = run_fixed_weight_study(
            frame,
            args.cutoff,
            args.n_splits,
            args.n_trials,
            scale_pos_weight=weight,
            oversample=True,
            track=True,
        )
        write_best_params(RESAMPLE_PARAMS, resampled)
        log_search_result(resampled)
    print(f"resample mean F1 {resampled['best_mean_f1']:.4f} test F1 {resampled['test_f1']:.4f}")


if __name__ == "__main__":
    main()
