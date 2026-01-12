.PHONY: help install test clean train tensorboard format lint

help:
	@echo "RoboGym - Available Commands"
	@echo "============================"
	@echo "make install      - Install all dependencies"
	@echo "make test         - Verify installation"
	@echo "make train        - Start training pipeline"
	@echo "make tensorboard  - Start TensorBoard"
	@echo "make clean        - Remove generated files"
	@echo "make format       - Format code with black"
	@echo "make lint         - Lint code with ruff"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -e .
	@echo "✓ Installation complete"

test:
	@echo "Running installation tests..."
	python test_installation.py

train:
	@echo "Starting training pipeline..."
	python main.py

tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006"
	tensorboard --logdir=./runs

clean:
	@echo "Cleaning generated files..."
	rm -rf runs/* log/* hyperparameter_tuning/optuna_studies/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "✓ Cleanup complete"

format:
	@echo "Formatting code..."
	pip install black
	black agents/ environments/ utils/ main.py
	@echo "✓ Formatting complete"

lint:
	@echo "Linting code..."
	pip install ruff
	ruff check agents/ environments/ utils/ main.py
	@echo "✓ Linting complete"
