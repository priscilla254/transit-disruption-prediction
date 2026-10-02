"""Slice the shipped model's November–December predictions.

The hyperparameters are the unweighted baseline set. The cutoff is the operating
threshold already chosen on January–October. The holdout is scored once, then
grouped. Weather bands exist only for these slices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from sbahn.models.evaluate import BASELINE_PARAMS, OPERATING_PATH
from sbahn.models.splits import temporal_split
from sbahn.models.tune import _fit_fold, write_best_params

PREDICTIONS_PATH = Path("reports/evaluation/test_predictions.parquet")
SLICES_PATH = Path("reports/evaluation/slices.json")
HEAVY_MM = 2.0
WEATHER_LEVELS = ("none", "light", "heavy")
DISRUPTION_LEVELS = ("strike", "incident", "event", "none")


def weather_band(precipitation: pd.Series) -> pd.Series:
    """Bucket millimetres into none, light (through 2 mm), and heavy."""
    values = precipitation.astype(float)
    band = pd.Series("heavy", index=values.index, dtype="object")
    band = band.mask(values <= 0, "none")
    band = band.mask((values > 0) & (values <= HEAVY_MM), "light")
    return band


def disruption_label(frame: pd.DataFrame) -> pd.Series:
    """One label per trip: strike, then incident, then event, otherwise none."""
    label = pd.Series("none", index=frame.index, dtype="object")
    label = label.mask(frame["is_event_day"].astype(int).eq(1), "event")
    label = label.mask(frame["has_incident"].astype(int).eq(1), "incident")
    label = label.mask(frame["is_strike_day"].astype(int).eq(1), "strike")
    return label


def group_metrics(frame: pd.DataFrame, column: str, levels: list | tuple) -> list[dict]:
    """Delayed-class precision, recall, and F1 for each group, including empty ones."""
    rows: list[dict] = []
    for level in levels:
        part = frame.loc[frame[column] == level]
        n = len(part)
        group = int(level) if isinstance(level, int) else str(level)
        if n == 0:
            rows.append(
                {
                    "group": group,
                    "n": 0,
                    "delay_rate": None,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
            )
            continue
        labels = part["is_delayed"].astype(int)
        predicted = part["predicted"].astype(int)
        rows.append(
            {
                "group": group,
                "n": n,
                "delay_rate": float(labels.mean()),
                "precision": float(precision_score(labels, predicted, zero_division=0)),
                "recall": float(recall_score(labels, predicted, zero_division=0)),
                "f1": float(f1_score(labels, predicted, zero_division=0)),
            }
        )
    return rows


def slice_tables(frame: pd.DataFrame) -> dict[str, list[dict]]:
    hours = sorted(int(hour) for hour in frame["hour"].unique())
    lines = sorted(str(line) for line in frame["line_id"].unique())
    return {
        "line_id": group_metrics(frame, "line_id", lines),
        "hour": group_metrics(frame, "hour", hours),
        "weather_band": group_metrics(frame, "weather_band", WEATHER_LEVELS),
        "disruption": group_metrics(frame, "disruption", DISRUPTION_LEVELS),
    }


def predict_holdout(
    features: pd.DataFrame,
    params: dict[str, float | int],
    cutoff: str,
    threshold: float,
) -> pd.DataFrame:
    """Fit once on January–October and score November–December once."""
    development, test = temporal_split(features, "scheduled_departure_time", cutoff)
    if test["scheduled_departure_time"].min() < pd.Timestamp(cutoff):
        raise AssertionError("holdout includes a row before the cutoff")
    model, test_x = _fit_fold(development, test, params)
    probabilities = model.predict_proba(test_x)[:, 1]
    out = pd.DataFrame(
        {
            "is_delayed": test["is_delayed"].astype(int).to_numpy(),
            "probability": probabilities,
            "predicted": (probabilities >= threshold).astype(int),
            "line_id": test["line_id"].astype(str).to_numpy(),
            "hour": test["hour"].astype(int).to_numpy(),
            "precipitation_mm": test["precipitation_mm"].astype(float).to_numpy(),
            "has_incident": test["has_incident"].astype(int).to_numpy(),
            "is_event_day": test["is_event_day"].astype(int).to_numpy(),
            "is_strike_day": test["is_strike_day"].astype(int).to_numpy(),
        }
    )
    out["weather_band"] = weather_band(out["precipitation_mm"])
    out["disruption"] = disruption_label(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice the shipped model on the holdout.")
    parser.add_argument("--in", dest="in_path", default="data/processed/trips_features.parquet")
    parser.add_argument("--params", default=str(BASELINE_PARAMS))
    parser.add_argument("--threshold-file", default=str(OPERATING_PATH))
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH))
    parser.add_argument("--out", default=str(SLICES_PATH))
    args = parser.parse_args()
    saved = json.loads(Path(args.params).read_text(encoding="utf-8"))
    operating = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
    threshold = float(operating["threshold"])
    features = pd.read_parquet(args.in_path)
    predictions = predict_holdout(
        features,
        saved["best_params"],
        cutoff=str(operating["cutoff"]),
        threshold=threshold,
    )
    prediction_path = Path(args.predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)
    tables = slice_tables(predictions)
    payload = {
        "cutoff": operating["cutoff"],
        "threshold": threshold,
        "model": "no-weighting",
        "test_rows": len(predictions),
        "slices": tables,
    }
    write_best_params(Path(args.out), payload)
    print(f"wrote {len(predictions):,} predictions at threshold {threshold:.2f}")
    for name, rows in tables.items():
        print(name)
        for row in rows:
            rate = "n/a" if row["delay_rate"] is None else f"{row['delay_rate']:.3f}"
            print(
                f"  {row['group']}: n {row['n']}  delay rate {rate}  "
                f"precision {row['precision']:.3f}  recall {row['recall']:.3f}  "
                f"F1 {row['f1']:.3f}"
            )


if __name__ == "__main__":
    main()
