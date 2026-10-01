.PHONY: install test interim data features tune track evaluate

install:
	python -m pip install -e ".[dev,notebooks]"

test:
	python -m pytest

interim:
	python -m sbahn.data.interim

data:
	python -m sbahn.data.build_dataset --raw-dir data/raw --out data/interim/trips_merged.parquet

features:
	python -m sbahn.features.build --in data/interim/trips_merged.parquet --out data/processed/trips_features.parquet

tune:
	python -m sbahn.models.tune --in data/processed/trips_features.parquet --cutoff 2024-11-01 --n-splits 5 --n-trials 40

track:
	python -m sbahn.models.tracking

evaluate:
	python -m sbahn.models.evaluate
