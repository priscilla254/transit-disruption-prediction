"""Explain the shipped model with TreeSHAP on the November–December holdout.

The trees are the unweighted baseline, fit once on January–October. SHAP values
are log-odds of delay: the expected value plus one row's values equals that
row's model margin. The operating threshold only labels a trip delayed or on
time. It does not enter the explanation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sbahn.models.evaluate import BASELINE_PARAMS, OPERATING_PATH
from sbahn.models.slices import disruption_label
from sbahn.models.splits import temporal_split
from sbahn.models.tune import _fit_fold, write_best_params

IMPORTANCE_PATH = Path("reports/evaluation/shap_importance.json")
CASES_PATH = Path("reports/evaluation/shap_cases.json")
FIGURES = Path("reports/figures")
TOP_FEATURES = 15
CASE_ROLES = (
    "strike_caught",
    "ordinary_miss",
    "ordinary_false_alarm",
    "s41_miss",
)


def tree_explanation(model, frame: pd.DataFrame) -> shap.Explanation:
    """TreeSHAP for every row. No background sample and no second fit."""
    return shap.TreeExplainer(model)(frame)


def mean_abs_importance(explanation: shap.Explanation) -> list[dict]:
    """Rank features by mean absolute SHAP on the explained rows."""
    scores = np.abs(np.asarray(explanation.values)).mean(axis=0)
    order = np.argsort(-scores, kind="mergesort")
    names = list(explanation.feature_names)
    rows = []
    for rank, position in enumerate(order, start=1):
        rows.append(
            {
                "name": names[int(position)],
                "mean_abs_shap": float(scores[int(position)]),
                "rank": rank,
            }
        )
    return rows


def select_cases(frame: pd.DataFrame, threshold: float) -> dict[str, int]:
    """Row positions for the four holdout stories.

    Closest means the smallest absolute distance from the operating threshold.
    A tie keeps the earlier row.
    """
    probability = frame["probability"].to_numpy(dtype=float)
    delayed = frame["is_delayed"].astype(int).eq(1).to_numpy()
    predicted = frame["predicted"].astype(int).eq(1).to_numpy()
    disruption = frame["disruption"].astype(str).to_numpy()
    line = frame["line_id"].astype(str).to_numpy()
    masks = {
        "strike_caught": (disruption == "strike") & delayed & predicted,
        "ordinary_miss": (disruption == "none") & delayed & ~predicted,
        "ordinary_false_alarm": (disruption == "none") & ~delayed & predicted,
        "s41_miss": (line == "S41") & delayed & ~predicted,
    }
    chosen: dict[str, int] = {}
    for role in CASE_ROLES:
        candidates = np.flatnonzero(masks[role])
        if len(candidates) == 0:
            raise ValueError(f"no holdout row matches {role}")
        distance = np.abs(probability[candidates] - threshold)
        order = np.lexsort((candidates, distance))
        chosen[role] = int(candidates[int(order[0])])
    return chosen


def _feature_value(value) -> str | int | float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def case_records(
    explanation: shap.Explanation,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    positions: dict[str, int],
) -> list[dict]:
    """One record per role, with every feature's SHAP contribution."""
    records = []
    names = list(explanation.feature_names)
    for role, position in positions.items():
        values = np.asarray(explanation.values[position], dtype=float)
        order = np.argsort(-np.abs(values), kind="mergesort")
        row = frame.iloc[position]
        feature_row = features.iloc[position]
        records.append(
            {
                "role": role,
                "row": int(position),
                "line_id": str(row["line_id"]),
                "hour": int(row["hour"]),
                "disruption": str(row["disruption"]),
                "probability": float(row["probability"]),
                "predicted": int(row["predicted"]),
                "is_delayed": int(row["is_delayed"]),
                "contributions": [
                    {
                        "feature": names[int(index)],
                        "value": _feature_value(feature_row[names[int(index)]]),
                        "shap": float(values[int(index)]),
                    }
                    for index in order
                ],
            }
        )
    return records


def _display_explanation(
    explanation: shap.Explanation, features: pd.DataFrame, position: int
) -> shap.Explanation:
    """One row, with category labels instead of integer codes."""
    row = explanation[position]
    data = [_feature_value(features.iloc[position][name]) for name in row.feature_names]
    return shap.Explanation(
        values=np.asarray(row.values, dtype=float),
        base_values=float(np.asarray(row.base_values).reshape(-1)[0]),
        data=np.array(data, dtype=object),
        feature_names=list(row.feature_names),
    )


def save_global_plots(explanation: shap.Explanation, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    shap.plots.bar(explanation, max_display=TOP_FEATURES, show=False)
    plt.savefig(figure_dir / "shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    shap.plots.beeswarm(explanation, max_display=TOP_FEATURES, show=False)
    plt.savefig(figure_dir / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()


def save_waterfalls(
    explanation: shap.Explanation,
    features: pd.DataFrame,
    positions: dict[str, int],
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for role, position in positions.items():
        shap.plots.waterfall(
            _display_explanation(explanation, features, position),
            max_display=TOP_FEATURES,
            show=False,
        )
        plt.savefig(figure_dir / f"shap_waterfall_{role}.png", dpi=120, bbox_inches="tight")
        plt.close()


def explain_holdout(
    features: pd.DataFrame,
    params: dict[str, float | int],
    cutoff: str,
    threshold: float,
) -> tuple[shap.Explanation, pd.DataFrame, pd.DataFrame]:
    """Fit once and explain every November–December row."""
    development, test = temporal_split(features, "scheduled_departure_time", cutoff)
    if test["scheduled_departure_time"].min() < pd.Timestamp(cutoff):
        raise AssertionError("holdout includes a row before the cutoff")
    model, test_x = _fit_fold(development, test, params)
    probabilities = model.predict_proba(test_x)[:, 1]
    meta = pd.DataFrame(
        {
            "is_delayed": test["is_delayed"].astype(int).to_numpy(),
            "probability": probabilities,
            "predicted": (probabilities >= threshold).astype(int),
            "line_id": test["line_id"].astype(str).to_numpy(),
            "hour": test["hour"].astype(int).to_numpy(),
            "disruption": disruption_label(test).to_numpy(),
        }
    )
    return tree_explanation(model, test_x), meta, test_x.reset_index(drop=True)


def log_shap_run(importance: list[dict], cases: list[dict], cutoff: str, threshold: float) -> None:
    import mlflow

    from sbahn.models.tracking import study_run

    top = ",".join(row["name"] for row in importance[:10])
    with study_run("shap", {"treatment": "shap", "source": "live"}):
        mlflow.log_params(
            {
                "cutoff": cutoff,
                "threshold": threshold,
                "model": "no-weighting",
                "top_features": top,
            }
        )
        for case in cases:
            mlflow.log_metric(f"{case['role']}_probability", float(case["probability"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain the shipped model with TreeSHAP.")
    parser.add_argument("--in", dest="in_path", default="data/processed/trips_features.parquet")
    parser.add_argument("--params", default=str(BASELINE_PARAMS))
    parser.add_argument("--threshold-file", default=str(OPERATING_PATH))
    parser.add_argument("--importance", default=str(IMPORTANCE_PATH))
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--figures", default=str(FIGURES))
    args = parser.parse_args()
    saved = json.loads(Path(args.params).read_text(encoding="utf-8"))
    operating = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
    threshold = float(operating["threshold"])
    features = pd.read_parquet(args.in_path)
    explanation, meta, test_x = explain_holdout(
        features,
        saved["best_params"],
        cutoff=str(operating["cutoff"]),
        threshold=threshold,
    )
    importance = mean_abs_importance(explanation)
    positions = select_cases(meta, threshold)
    cases = case_records(explanation, meta, test_x, positions)
    figure_dir = Path(args.figures)
    save_global_plots(explanation, figure_dir)
    save_waterfalls(explanation, test_x, positions, figure_dir)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    write_best_params(
        Path(args.importance),
        {
            "cutoff": operating["cutoff"],
            "threshold": threshold,
            "model": "no-weighting",
            "test_rows": int(len(meta)),
            "features": importance,
        },
    )
    write_best_params(
        Path(args.cases),
        {
            "cutoff": operating["cutoff"],
            "threshold": threshold,
            "model": "no-weighting",
            "base_value": base_value,
            "cases": cases,
        },
    )
    log_shap_run(importance, cases, cutoff=str(operating["cutoff"]), threshold=threshold)
    print(f"explained {len(meta):,} holdout rows")
    for row in importance[:10]:
        print(f"  {row['rank']:2d}  {row['name']}  {row['mean_abs_shap']:.4f}")
    for case in cases:
        lead = case["contributions"][0]
        print(
            f"{case['role']}: {case['line_id']} hour {case['hour']}  "
            f"probability {case['probability']:.3f}  "
            f"leading {lead['feature']} {lead['shap']:+.3f}"
        )


if __name__ == "__main__":
    main()
