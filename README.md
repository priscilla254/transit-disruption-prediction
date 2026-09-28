# Transit disruption prediction

**Synthetic data.** The Berlin S-Bahn punctuality records used in this project are synthetic. They are not official S-Bahn Berlin or Deutsche Bahn operational data. Delays, cancellations, incidents, weather, events, and strikes in this dataset do not describe real service. Do not use results from this project to judge actual punctuality or to make operational decisions.

This repository is a placeholder for exploring whether trip, weather, and disruption features can predict S-Bahn delay and cancellation. Work lives in `notebooks/transit_disruption_prediction.ipynb`.

```
data/       local dataset files (not committed)
notebooks/  analysis notebooks
src/        project code
tests/      tests
docs/       notes
reports/    figures and written results
```

## Data

[Berlin S-Bahn Punctuality Database](https://www.kaggle.com/datasets/alperenmyung/berlin-s-bahn-punctuality-database) on Kaggle (`alperenmyung/berlin-s-bahn-punctuality-database`).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install kagglehub pandas
```
