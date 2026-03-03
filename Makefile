# Makefile for LLM Fine-Tuning with PEFT

.PHONY: help install install-dev setup clean prepare-data download-model validate train evaluate deploy test format lint

help:
	@echo "Available commands:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make setup            - Complete setup (install + create directories)"
	@echo "  make clean            - Clean generated files"
	@echo "  make prepare-data     - Prepare dataset for training"
	@echo "  make download-model   - Download base model"
	@echo "  make validate         - Validate dataset quality"
	@echo "  make train            - Run LoRA fine-tuning"
	@echo "  make evaluate         - Evaluate trained model"
	@echo "  make deploy           - Deploy model to production"
	@echo "  make test             - Run tests"
	@echo "  make format           - Format code with black and isort"
	@echo "  make lint             - Run linters"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

setup: install
	mkdir -p data/raw data/processed data/splits
	mkdir -p models outputs/checkpoints outputs/results outputs/logs
	mkdir -p logs notebooks tests
	cp .env.example .env || true
	@echo "Setup complete! Edit .env file with your configuration."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf dist build
	@echo "Cleaned generated files"

prepare-data:
	python scripts/prepare_data.py \
		--output-dir data/splits \
		--num-examples 10000 \
		--config configs/data_config.yaml

download-model:
	python scripts/download_model.py \
		--model-name llama-2-7b \
		--output-dir models

validate:
	python scripts/validate_dataset.py \
		--dataset-path data/splits/train.jsonl \
		--output-path data/validation_report.json

train:
	python src/training/train_lora.py \
		--config configs/lora_config.yaml

evaluate:
	python src/evaluation/evaluate.py \
		--model-path outputs/final \
		--test-data data/splits/test.jsonl

deploy:
	python src/deployment/merge_and_serve.py \
		--base-model models/llama-2-7b \
		--lora-weights outputs/final

test:
	pytest tests/ -v --cov=src --cov-report=html

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/
	pylint src/ scripts/ tests/
	mypy src/ scripts/ tests/

.DEFAULT_GOAL := help