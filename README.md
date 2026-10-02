# Transit disruption prediction

**Synthetic data.** The Berlin S-Bahn punctuality records used in this project are synthetic. They are not official S-Bahn Berlin or Deutsche Bahn operational data. Delays, cancellations, incidents, weather, events, and strikes in this dataset do not describe real service. Do not use results from this project to judge actual punctuality or to make operational decisions.

Trip, weather, and disruption features are used to predict whether a trip is delayed. The exploratory checks live in `notebooks/01_eda.ipynb`. The evaluation is summarized below and written up in `docs/model_card.md`.

```
data/raw/          Kaggle files, not committed
data/interim/      validated Parquet
data/processed/    model-ready feature tables
notebooks/         01_eda, 02_features, 03_evaluation
src/sbahn/         loading, features, models, predict
tests/             joins, leakage, temporal split
docs/              data quality, feature dictionary, model card
reports/           evaluation tables and figures
```

## Data summary

Source: [Berlin S-Bahn Punctuality Database](https://www.kaggle.com/datasets/alperenmyung/berlin-s-bahn-punctuality-database) (`alperenmyung/berlin-s-bahn-punctuality-database`). Counts below come from the SQLite file, checked in `notebooks/01_eda.ipynb`.

- 131,771 trips, from 2024-01-01 through 2024-12-30 (365 dates). 2024-12-31 has no trips.
- Target: `is_delayed` is 1 exactly when `delay_minutes` is 6 or more. Minutes 0–5 are on time.
- 860 trips (0.65%) are cancelled, and they sit in both delay classes (65 on time, 795 delayed). They are dropped for this use case, so `is_cancelled` is a row filter, not a feature. See `docs/feature_dictionary.md`.
- After that filter, 126,968 trips are on time and 3,943 are delayed (32.2 to 1). A model that always predicts on time would score about 97% accuracy, so accuracy is a poor summary of this label.
- `delay_minutes` is heavy-tailed, including a long tail past an hour. The EDA histogram uses `log1p(delay_minutes)` so the zeros stay on the axis.
- Disruption tables are small and uneven: 36 incidents, 3 city events, and 3 strikes, all inside the trip calendar. June has no incidents. Exact dates are in `docs/data_quality.md`. An event-day feature will rest on only three dates.

## EDA

See `notebooks/01_eda.ipynb`. The 860 cancelled trips are dropped. About 3% of the remaining trips are delayed, and 0–5 minutes count as on time. Table checks and the leakage rules are in `docs/data_quality.md` and `docs/feature_dictionary.md`.

## Results

The shipped model is the unweighted LightGBM classifier in `reports/optuna/baseline_params.json`. It is fit on January–October (109,485 trips) and scored once on November–December (21,426 trips). A trip is called delayed when its probability is at least 0.30. That cutoff was chosen on January–October only. The full account, including the slice tables and the SHAP reading, is in `docs/model_card.md`. The plots are shown in `notebooks/03_evaluation.ipynb`.

At 0.30 the holdout confusion matrix is 20,380 true on-time, 162 false alarms, 91 missed delays, and 793 caught delays. Delayed-class F1 is 0.862, on-time recall is 0.992, and delayed recall is 0.897.

`reports/evaluation/test_metrics.json` scores the same trees at the search's own cutoff of 0.50 instead: delayed-class F1 0.874, macro-F1 0.934. That file also reports a separate regression model — the same tree settings, without `scale_pos_weight`, predicting `delay_minutes` instead of `is_delayed` — at MAE 1.65 minutes and RMSE 10.83 minutes. RMSE is much larger than MAE because a few large misses, likely the strike-day delays, are squared and dominate the average; most trips are predicted within a minute or two.

The 0.862 score is lifted by strike days. Ordinary days, with no strike, incident, or event, have delayed-class F1 0.717 and hold 126 of the 162 false alarms and 86 of the 91 misses. Strike days (565 trips, 93.6% delayed) have F1 0.962. S41 and S42 together account for 83 false alarms and 60 missed delays. The weakest hours are 08:00 (F1 0.707) and 16:00 (F1 0.703). Heavy rain, above 2 mm, holds 55 of the 91 missed delays. The event-day slice has no November–December trips, and the 8 incident trips were all on time.

TreeSHAP on those same rows starts from an expected log-odds of −6.77, about a 0.1% chance of delay, so a trip needs a large positive push to reach probability 0.30. The largest mean absolute contributions are the previous hour's mean delay on the line (0.426), weather condition (0.373), route (0.330), and the previous day's mean delay (0.259). A strike day is a large push when the flag is on. The four trips closest to 0.30 are in `reports/evaluation/shap_cases.json`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,notebooks]"
```

The Kaggle files are not committed. Download them and copy the extract into `data/raw/`:

```powershell
python -c "import kagglehub; print(kagglehub.dataset_download('alperenmyung/berlin-s-bahn-punctuality-database'))"
Copy-Item "$env:USERPROFILE\.cache\kagglehub\datasets\alperenmyung\berlin-s-bahn-punctuality-database\versions\2\*" data\raw\
```

`sbahn.config.database_path()` uses `data/raw/berlin_sbahn_delays.db` when that file is present, and otherwise the kagglehub cache.

Validated copies of the seven tables, with timestamps parsed, are written to `data/interim/` by:

```powershell
python -m sbahn.data.interim
```

Tuning runs are written to a local MLflow store at `mlruns/mlflow.db` (not committed). Record the studies already saved under `reports/optuna/`, then open the UI:

```powershell
python -m sbahn.models.tracking
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

`python -m sbahn.models.tune` and `python -m sbahn.models.compare_imbalance` record a new parent run for each treatment and a nested run for each Optuna trial. `python -m sbahn.models.evaluate` scores the unweighted model once on November–December and writes `reports/evaluation/test_metrics.json`.
