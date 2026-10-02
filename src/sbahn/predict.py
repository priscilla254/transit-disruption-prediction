"""Score a batch of trips with the shipped delay model.

The trees are the unweighted baseline, fit once on departures before the
operating cutoff. Rows at or after that cutoff are scored once. A trip is
called delayed when its probability is at least the operating threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sbahn.models.evaluate import BASELINE_PARAMS, OPERATING_PATH
from sbahn.models.splits import temporal_split
from sbahn.models.tune import _fit_fold


def score_batch(
    features: pd.DataFrame,
    params: dict[str, float | int],
    cutoff: str,
    threshold: float,
) -> pd.DataFrame:
    """Fit once on earlier departures and score every later row, in input order."""
    development, scored = temporal_split(features, "scheduled_departure_time", cutoff)
    if development.empty:
        raise ValueError("training set is empty")
    if scored.empty:
        raise ValueError("scored set is empty")
    model, scored_x = _fit_fold(development, scored, params)
    probabilities = model.predict_proba(scored_x)[:, 1]
    return pd.DataFrame(
        {
            "trip_id": scored["trip_id"].to_numpy(),
            "probability": probabilities,
            "predicted": (probabilities >= threshold).astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score trips with the shipped delay model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--params", default=str(BASELINE_PARAMS))
    parser.add_argument("--threshold-file", default=str(OPERATING_PATH))
    args = parser.parse_args()
    saved = json.loads(Path(args.params).read_text(encoding="utf-8"))
    operating = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
    threshold = float(operating["threshold"])
    predictions = score_batch(
        pd.read_parquet(args.input),
        saved["best_params"],
        cutoff=str(operating["cutoff"]),
        threshold=threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output, index=False)
    print(f"wrote {len(predictions):,} predictions at threshold {threshold:.2f}")


if __name__ == "__main__":
    main()
