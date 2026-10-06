"""Uncertainty for the shipped holdout score and the strike delay rate.

The delayed-class F1 interval resamples the scored November–December rows.
The two-proportion test compares actual delay rates on strike days and
ordinary days. Neither step refits the trees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score

from sbahn.models.tune import SEED, write_best_params

PREDICTIONS_PATH = Path("reports/evaluation/test_predictions.parquet")
OUT_PATH = Path("reports/evaluation/uncertainty.json")
N_RESAMPLES = 1000


def delayed_f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(labels, predicted, zero_division=0))


def bootstrap_f1_interval(
    labels: np.ndarray,
    predicted: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict:
    """95% percentile interval for delayed-class F1 on resampled rows."""
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    n = len(labels)
    if n == 0:
        raise ValueError("holdout is empty")
    point = delayed_f1(labels, predicted)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        draw = rng.integers(0, n, size=n)
        scores[index] = delayed_f1(labels[draw], predicted[draw])
    low, high = np.quantile(scores, [0.025, 0.975])
    return {
        "f1": point,
        "n": int(n),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "level": 0.95,
        "low": float(low),
        "high": float(high),
    }


def two_proportion_ztest(
    successes_a: int, n_a: int, successes_b: int, n_b: int
) -> dict[str, float]:
    """Two-sided pooled z-test of rate A against rate B."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both groups need rows")
    rate_a = successes_a / n_a
    rate_b = successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    variance = pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)
    difference = rate_a - rate_b
    if variance == 0.0:
        z = 0.0 if difference == 0.0 else float("inf")
        p_value = 1.0 if difference == 0.0 else 0.0
    else:
        z = difference / float(np.sqrt(variance))
        p_value = float(2.0 * norm.sf(abs(z)))
    return {
        "rate_a": float(rate_a),
        "rate_b": float(rate_b),
        "rate_difference": float(difference),
        "z": float(z),
        "p_value": p_value,
    }


def delay_rate_test(frame: pd.DataFrame) -> dict:
    """Compare the strike-day delay rate with the ordinary-day delay rate."""
    strike = frame.loc[frame["disruption"].astype(str).eq("strike"), "is_delayed"].astype(int)
    ordinary = frame.loc[frame["disruption"].astype(str).eq("none"), "is_delayed"].astype(int)
    delayed_strike = int(strike.sum())
    delayed_ordinary = int(ordinary.sum())
    tested = two_proportion_ztest(delayed_strike, len(strike), delayed_ordinary, len(ordinary))
    return {
        "strike": {
            "n": len(strike),
            "delayed": delayed_strike,
            "rate": tested["rate_a"],
        },
        "ordinary": {
            "n": len(ordinary),
            "delayed": delayed_ordinary,
            "rate": tested["rate_b"],
        },
        "rate_difference": tested["rate_difference"],
        "z": tested["z"],
        "p_value": tested["p_value"],
    }


def uncertainty_report(
    frame: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> dict:
    labels = frame["is_delayed"].astype(int).to_numpy()
    predicted = frame["predicted"].astype(int).to_numpy()
    return {
        "holdout_delayed_f1": bootstrap_f1_interval(
            labels, predicted, n_resamples=n_resamples, seed=seed
        ),
        "delay_rate_test": delay_rate_test(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the holdout F1 and test strike versus ordinary delay rates."
    )
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    predictions = Path(args.predictions)
    if not predictions.is_file():
        raise FileNotFoundError(
            f"{predictions} is missing. Run python -m sbahn.models.slices first."
        )
    payload = uncertainty_report(
        pd.read_parquet(predictions), n_resamples=args.n_resamples, seed=args.seed
    )
    write_best_params(Path(args.out), payload)
    interval = payload["holdout_delayed_f1"]
    tested = payload["delay_rate_test"]
    print(
        f"delayed-class F1 {interval['f1']:.4f}  "
        f"95% interval {interval['low']:.4f} to {interval['high']:.4f}"
    )
    print(
        f"strike rate {tested['strike']['rate']:.4f}  "
        f"ordinary rate {tested['ordinary']['rate']:.4f}  "
        f"z {tested['z']:.2f}  p {tested['p_value']:.3g}"
    )


if __name__ == "__main__":
    main()
