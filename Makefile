.PHONY: install lint test interim data features train tune track evaluate slices shap predict uncertainty pipeline

install:
	python -m pip install -e ".[dev,notebooks]"

lint:
	python -m ruff check .

test:
	python -m pytest

interim:
	python -m sbahn.data.interim

data:
	python -m sbahn.data.build_dataset --raw-dir data/raw --out data/interim/trips_merged.parquet

features:
	python -m sbahn.features.build --in data/interim/trips_merged.parquet --out data/processed/trips_features.parquet

train tune:
	python -m sbahn.models.tune --in data/processed/trips_features.parquet --cutoff 2024-11-01 --n-splits 5 --n-trials 40

track:
	python -m sbahn.models.tracking

evaluate:
	python -m sbahn.models.evaluate

slices:
	python -m sbahn.models.slices

shap:
	python -m sbahn.models.shap_explain

predict:
	python -m sbahn.predict --input data/processed/trips_features.parquet --output reports/evaluation/batch_predictions.parquet

uncertainty:
	python -m sbahn.models.uncertainty

pipeline:
	$(MAKE) interim
	$(MAKE) data
	$(MAKE) features
	$(MAKE) train
	$(MAKE) evaluate
	$(MAKE) slices
	$(MAKE) shap
