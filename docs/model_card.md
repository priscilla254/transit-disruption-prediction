# Model card

A LightGBM classifier was tuned with Optuna. The records are synthetic. They are not a description of real S-Bahn Berlin service.

## Intended use

Estimate whether a Berlin S-Bahn trip will be delayed, using only values that would already be known on the platform before `scheduled_departure_time`.

## Target

`is_delayed` is 1 exactly when `delay_minutes` is 6 or more. Cancelled trips are dropped before modeling, so `is_cancelled` is a row filter rather than a second target.

## Training data

Features come from `data/processed/trips_features.parquet`. Inputs are the columns in `FEATURE_COLUMNS`, except `scheduled_departure_time`, `line_name`, `start_name`, and `end_name`. Columns that must not be used as inputs are listed in `docs/feature_dictionary.md`.

The search uses trips before 2024-11-01 (January–October): 109,485 rows, delayed rate 2.79%. `TimeSeriesSplit` with 5 folds walks forward through that block. Optuna ran 40 trials and maximized mean F1 of the delayed class at the 0.5 threshold. November and December were not part of any trial.

The best development mean F1 is 0.379. Those parameters are in `reports/optuna/best_params.json`. The chosen `scale_pos_weight` is 22.2.

Refitting that one parameter set on the five development folds does not produce a rising F1. `TimeSeriesSplit` cuts by row count, so each validation block is a different part of the year. The two strike windows inside January–October land in fold 1 and fold 3.

| Fold | Validation window | Delayed rate | Strike trips | F1 |
|---|---|---:|---:|---:|
| 1 | 2024-02-20 to 2024-04-10 | 4.9% | 577 | 0.040 |
| 2 | 2024-04-10 to 2024-05-31 | 0.8% | 0 | 0.291 |
| 3 | 2024-05-31 to 2024-07-22 | 8.8% | 1,381 | 0.898 |
| 4 | 2024-07-22 to 2024-09-10 | 1.0% | 0 | 0.400 |
| 5 | 2024-09-10 to 2024-10-31 | 0.5% | 0 | 0.264 |

Fold 1 trains only through 2024-02-20, before any strike, so `is_strike_day` is always 0 in that training block. Its validation window then contains the March strike (2024-03-15 to 2024-03-17). Fold 3 validates across the June strike (2024-06-20 to 2024-06-25) after the March strike is already in the training rows, and F1 jumps to 0.898. Folds 2, 4, and 5 have no strike in the validation window, and the delayed rate falls from 0.8% to 0.5%, so a longer training prefix does not raise F1. The mean of 0.379 mixes those two kinds of month.

## Test score

The same parameter set, fit on all of January–October, scores F1 0.818 on 2024-11-01 through 2024-12-30 (21,426 rows, delayed rate 4.13%, precision 0.748, recall 0.903). That number was not used to pick another trial.

The November strike window is a large part of the lift. On the 565 strike-day test trips the delayed rate is 93.6% and F1 is 0.950. On the other test trips the delayed rate is 1.70% and F1 is 0.658.

## Imbalance comparison

Four treatments share the same January–October search and the same November–December holdout. Development F1 for the Optuna rows is the mean of the five fold F1s. The threshold row is different: one F1 on the stitched out-of-fold predictions. At 0.5 that pooled score is 0.634, while the mean of the fold F1s for the same weighted model stays 0.379.

| Method | Development F1 | Test F1 | Threshold |
|---|---:|---:|---:|
| No weighting | 0.297 | 0.874 | 0.5 |
| Class weights | 0.379 | 0.818 | 0.5 |
| Class weights, threshold sweep | 0.634 | 0.818 | 0.5 |
| Class weights plus resampling | 0.435 | 0.734 | 0.5 |

Each treatment is also a run in the local MLflow store `mlruns/mlflow.db`, experiment `sbahn-delay`.

No weighting has the best test F1. The sweep did not move the cutoff: pooled development F1 peaks at 0.5, and the holdout score stays 0.818. Copying delayed rows inside each training fold, on top of `scale_pos_weight` 22.2, raises the development mean to 0.435 and lowers test F1 to 0.734. That matches the strike cold-start. Fold 1 has no strike to copy, so extra delayed rows cannot teach it. Once March and June are in the final training set, the unweighted model already scores the November strike at F1 0.957, and on the other test days it scores 0.740 against 0.658 with class weights and 0.533 with resampling.

## Final test metrics

The unweighted parameters in `reports/optuna/baseline_params.json` were fit once on all 109,485 January–October rows and scored once on the 21,426 November–December rows. The cutoff stays at 0.5. These scores were not used to choose another trial. They are in `reports/evaluation/test_metrics.json`.

| Metric | Score |
|---|---:|
| Macro-F1 | 0.934 |
| Recall, on time | 0.995 |
| Recall, delayed | 0.863 |
| Delayed-class F1 | 0.874 |
| MAE (minutes) | 1.65 |
| RMSE (minutes) | 10.83 |

Macro-F1 averages the on-time F1 and the delayed F1. The delayed-class F1 matches the unweighted holdout score above. MAE and RMSE come from a separate minute model: the same tree settings, with `scale_pos_weight` removed, predicting `delay_minutes` joined from `data/interim/trips_merged.parquet`. That column is the regression target, not a classifier input. RMSE is much larger than MAE because a few very long delays dominate the squared error.

## Operating threshold

The search scored every trial at 0.5. The operating cutoff for the unweighted model was chosen afterward, on pooled out-of-fold predictions from January–October only. The same grid as the earlier sweep was used: 0.10 to 0.90 in steps of 0.05, maximizing delayed-class F1. November–December was not part of that choice.

The development F1 peaks at 0.30 (0.647). The neighbor at 0.35 is 0.646, and 0.50 is 0.633, so the cutoff moves from 0.50 to 0.30. The full grid is in `reports/evaluation/operating_threshold.json`.

At 0.30 on November–December the confusion matrix is 20,380 true on-time, 162 false alarms, 91 missed delays, and 793 caught delays. Delayed recall is 0.897 and on-time recall is 0.992. Delayed-class F1 is 0.862, a little below the 0.874 scored at 0.50. That holdout gap is not a reason to move the cutoff back: the cutoff was locked on January–October.

The calibration curve uses ten equal-count bins of the predicted probability. Nine bins sit on the origin: the model assigns nearly zero probability to most trips, and those trips are almost all on time. The top bin has a mean predicted probability of 0.40 and a delayed share of 0.41, on the diagonal. Plots: `reports/figures/confusion_matrix.png` and `reports/figures/calibration_curve.png`.

## Metrics

Accuracy is a poor summary: after cancellations are dropped, on-time trips outnumber delayed trips by about 32 to 1. The search objective is F1.

## Limits

Disruption context is sparse: 36 incidents, 3 city events, and 3 strikes. An event-day feature rests on three dates. `delay_propensity` and `delay_impact_factor` are excluded because they can stand in for the delay label.
