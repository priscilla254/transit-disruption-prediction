.PHONY: install test interim data

install:
	python -m pip install -e ".[dev,notebooks]"

test:
	python -m pytest

interim:
	python -m sbahn.data.interim

data:
	python -m sbahn.data.build_dataset --raw-dir data/raw --out data/interim/trips_merged.parquet
