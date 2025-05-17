.PHONY: help clean data train evaluate pipeline demo lint test

help:
	@echo "Available commands:"
	@echo "  make help      - Show this help message"
	@echo "  make clean     - Remove generated files and directories"
	@echo "  make data      - Generate synthetic data"
	@echo "  make train     - Train the ranking model"
	@echo "  make evaluate  - Evaluate the trained model"
	@echo "  make pipeline  - Run the entire pipeline"
	@echo "  make demo      - Run the Jupyter notebook demo"
	@echo "  make lint      - Run linting tools"
	@echo "  make test      - Run tests"

clean:
	rm -rf data models reports
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

data:
	python -m src.data_generation.main

data-sample:
	python -m src.data_generation.main --sample

train:
	python -m src.modeling.train

train-sample:
	python -m src.modeling.train --sample

evaluate:
	python -m src.modeling.evaluation

pipeline:
	python run_pipeline.py

pipeline-sample:
	python run_pipeline.py --sample

demo:
	jupyter notebook hotel_ranking_demo.ipynb

lint:
	black src
	isort src
	flake8 src

test:
	pytest -xvs tests/
