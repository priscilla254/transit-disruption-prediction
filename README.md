# Transit disruption prediction

**Synthetic data.** The Berlin S-Bahn punctuality records used in this project are synthetic. They are not official S-Bahn Berlin or Deutsche Bahn operational data. Delays, cancellations, incidents, weather, events, and strikes in this dataset do not describe real service. Do not use results from this project to judge actual punctuality or to make operational decisions.

This repository is a placeholder for exploring whether trip, weather, and disruption features can predict S-Bahn delay. The exploratory checks live in `notebooks/01_eda.ipynb`.

```
data/raw/          Kaggle files, not committed
data/interim/      validated Parquet
data/processed/    model-ready feature tables
notebooks/         01_eda, 02_features, 03_evaluation
src/sbahn/         loading, features, models, predict
tests/             joins, leakage, temporal split
docs/              data quality, feature dictionary, model card
reports/figures/   saved figures
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
