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

## Error analysis by slice

The shipped rule — baseline hyperparameters, call delayed when the probability is at least 0.30 — was fit once on January–October and scored once on November–December. Those predictions are in `reports/evaluation/test_predictions.parquet`. The four groupings in `reports/evaluation/slices.json` reuse that table. Weather bands are for the slice only: none is exactly 0 mm, light is above 0 through 2 mm, and heavy is above 2 mm.

On the whole holdout the rule raises 162 false alarms and misses 91 delays. Those errors are concentrated. Ordinary days, with no strike, incident, or event, hold 126 of the false alarms and 86 of the misses. Delayed-class precision there is 0.681 and recall is 0.758, so F1 falls to 0.717 from the overall 0.862. Strike days hold the other 36 false alarms and only 5 misses. The delay rate on those 565 trips is 93.6%, recall is 0.991, and F1 is 0.962. The model calls 560 of the 565 strike trips delayed, so every on-time strike trip is a false alarm and five delayed strike trips are missed.

| Slice | Trips | Delay rate | Precision | Recall | F1 | False alarms | Missed delays |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary day | 20,853 | 1.7% | 0.681 | 0.758 | 0.717 | 126 | 86 |
| Strike | 565 | 93.6% | 0.936 | 0.991 | 0.962 | 36 | 5 |
| Incident | 8 | 0% | 0.000 | 0.000 | 0.000 | 0 | 0 |
| Event | 0 | — | — | — | — | 0 | 0 |

All 8 incident trips were on time, and all 8 were called on time. Delayed-class F1 is 0 because that slice has no delayed trip. The event-day slice has no November–December trips, so it has no errors to count.

By line, S41 and S42 hold most of the mistakes. Together they are about a third of the holdout and account for 83 of the 162 false alarms and 60 of the 91 missed delays. S41 is the weaker of the two: precision 0.760, recall 0.808, 44 false alarms and 33 misses. S1 is the strongest line, with 15 false alarms and 4 misses.

| Line | Trips | Delay rate | Precision | Recall | F1 | False alarms | Missed delays |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 3,617 | 4.4% | 0.911 | 0.975 | 0.942 | 15 | 4 |
| S2 | 3,623 | 3.1% | 0.844 | 0.920 | 0.880 | 19 | 9 |
| S41 | 3,561 | 4.8% | 0.760 | 0.808 | 0.783 | 44 | 33 |
| S42 | 3,564 | 4.9% | 0.790 | 0.845 | 0.817 | 39 | 27 |
| S5 | 3,516 | 3.9% | 0.881 | 0.926 | 0.903 | 17 | 10 |
| S7 | 3,545 | 3.7% | 0.816 | 0.939 | 0.873 | 28 | 8 |

By hour, the commute peaks fail in both directions. At 08:00 (898 trips) there are 14 false alarms and 10 missed delays, and F1 is 0.707. At 16:00 (875 trips) there are 15 false alarms and 12 missed delays, and F1 is 0.703. The hour with the most false alarms is 17:00, with 21, plus 8 misses (F1 0.752). Midnight has 16 false alarms and 5 misses (F1 0.753). Hours 10, 14, and 23 miss no delays. Hour 20 raises no false alarms.

Rain raises the delay rate and lowers recall. Dry trips (exactly 0 mm) have 36 false alarms and 14 misses. Heavy rain, above 2 mm, has 75 false alarms and 55 of the 91 missed delays, from 7,948 of the 21,426 trips.

| Band | Trips | Delay rate | Precision | Recall | F1 | False alarms | Missed delays |
|---|---:|---:|---:|---:|---:|---:|---:|
| None, 0 mm | 7,610 | 3.4% | 0.872 | 0.946 | 0.908 | 36 | 14 |
| Light, above 0 through 2 mm | 5,868 | 4.3% | 0.819 | 0.913 | 0.863 | 51 | 22 |
| Heavy, above 2 mm | 7,948 | 4.7% | 0.809 | 0.852 | 0.830 | 75 | 55 |

## SHAP

TreeSHAP was run once on those same November–December rows. The values are log-odds of delay, and they add up to the model margin. The expected log-odds is −6.77, so a trip starts as on time and needs a large positive push before its probability reaches 0.30. The ranking is in `reports/evaluation/shap_importance.json`. Plots: `reports/figures/shap_importance.png` and `reports/figures/shap_beeswarm.png`.

The largest mean absolute contributions are `line_mean_delay_prev_hour` (0.426), `weather_condition` (0.373), `route_key` (0.330), and `line_mean_delay_prev_day` (0.259). A high recent delay pushes the log-odds up, and so does a strike day. `is_strike_day` ranks fifth (0.126) because the flag is rare. When it is on, it is one of the strongest single pushes.

The four waterfalls are the holdout trips closest to 0.30 in each error story (`reports/evaluation/shap_cases.json`). The caught strike delay is an S41 trip at 08:00 with probability 0.304: `is_strike_day` adds +5.01 and yesterday's delay on the line adds more, while a quiet previous hour pulls back, so the trip sits just over the cutoff. The ordinary-day miss is an S42 trip at 09:00 with probability 0.279. Stormy weather, the peak flag, and a previous-hour mean delay of 3.5 minutes all push toward delay, and the sum still stays under 0.30. The ordinary-day false alarm is an S41 trip at 08:00 with probability 0.301, on time, pushed over the cutoff by stormy weather, wind, and that recent delay. The S41 miss is at 09:00 with probability 0.249: stormy weather and recent delay push it up, and its route pulls it back down.

## Metrics

Accuracy is a poor summary: after cancellations are dropped, on-time trips outnumber delayed trips by about 32 to 1. The search objective is F1.

## Limits

Disruption context is sparse: 36 incidents, 3 city events, and 3 strikes. An event-day feature rests on three dates. `delay_propensity` and `delay_impact_factor` are excluded because they can stand in for the delay label.
