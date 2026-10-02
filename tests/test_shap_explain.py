import math

import numpy as np
import pandas as pd

from sbahn.models.shap_explain import (
    mean_abs_importance,
    select_cases,
    tree_explanation,
)
from sbahn.models.splits import temporal_split
from sbahn.models.tune import _fit_fold

PARAMS = {
    "n_estimators": 8,
    "learning_rate": 0.1,
    "num_leaves": 4,
    "min_child_samples": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "scale_pos_weight": 1.0,
}


def _frame() -> pd.DataFrame:
    moments = list(pd.date_range("2024-01-01", periods=24, freq="7D"))
    moments += list(pd.date_range("2024-12-01", periods=6, freq="D"))
    rows = []
    for index, moment in enumerate(moments):
        rows.append(
            {
                "scheduled_departure_time": moment,
                "is_delayed": int(index % 3 == 0),
                "line_id": "S1" if index % 2 == 0 else "S41",
                "hour": int(index % 24),
                "is_strike_day": int(index % 5 == 0),
                "precipitation_mm": float(index % 4),
                "network_load": 3 + (index % 2),
            }
        )
    return pd.DataFrame(rows)


def test_shap_sums_to_the_margin():
    development, test = temporal_split(_frame(), "scheduled_departure_time", "2024-11-01")
    model, test_x = _fit_fold(development, test, PARAMS)
    explanation = tree_explanation(model, test_x)
    margin = model.predict(test_x, raw_score=True)
    reconstructed = np.asarray(explanation.base_values) + np.asarray(explanation.values).sum(axis=1)
    np.testing.assert_allclose(reconstructed, margin, atol=1e-5)
    importance = mean_abs_importance(explanation)
    assert len(importance) == test_x.shape[1]
    assert all(math.isfinite(row["mean_abs_shap"]) for row in importance)
    assert [row["rank"] for row in importance] == list(range(1, len(importance) + 1))
    scores = [row["mean_abs_shap"] for row in importance]
    assert scores == sorted(scores, reverse=True)


def test_case_picker_uses_distance_to_the_cutoff():
    frame = pd.DataFrame(
        {
            "disruption": ["strike", "none", "none", "none", "incident", "strike", "none"],
            "line_id": ["S5", "S1", "S2", "S41", "S41", "S7", "S1"],
            "is_delayed": [1, 1, 0, 1, 1, 1, 1],
            "predicted": [1, 0, 1, 0, 0, 1, 0],
            "probability": [0.95, 0.20, 0.31, 0.10, 0.28, 0.40, 0.40],
        }
    )
    chosen = select_cases(frame, 0.30)
    assert chosen == {
        "strike_caught": 5,
        "ordinary_miss": 1,
        "ordinary_false_alarm": 2,
        "s41_miss": 4,
    }
