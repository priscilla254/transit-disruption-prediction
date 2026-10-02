#!/bin/sh
set -eu

if [ ! -f data/raw/berlin_sbahn_delays.db ]; then
  echo "Mount the SQLite file at data/raw/berlin_sbahn_delays.db" >&2
  exit 1
fi

python -m sbahn.data.interim
python -m sbahn.data.build_dataset \
  --raw-dir data/raw \
  --out data/interim/trips_merged.parquet
python -m sbahn.features.build \
  --in data/interim/trips_merged.parquet \
  --out data/processed/trips_features.parquet
python -m sbahn.models.evaluate
python -m sbahn.models.slices
python -m sbahn.models.shap_explain
