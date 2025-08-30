# Coffee Law Makefile for common development tasks

.PHONY: help install install-dev test test-cov lint format type-check clean docs serve-docs run run-openai benchmark

# Default target
help:
	@echo "Coffee Law Development Commands"
	@echo "==============================="
	@echo "install       - Install the package in editable mode"
	@echo "install-dev   - Install with development dependencies"
	@echo "test          - Run the test suite"
	@echo "test-cov      - Run tests with coverage report"
	@echo "lint          - Run flake8 linting"
	@echo "format        - Format code with black and isort"
	@echo "type-check    - Run mypy type checking"
	@echo "clean         - Remove build artifacts and caches"
	@echo "docs          - Build documentation"
	@echo "serve-docs    - Serve documentation locally"
	@echo "run           - Run verification with mock embeddings"
	@echo "run-openai    - Run verification with OpenAI embeddings"
	@echo "benchmark     - Run performance benchmarks"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing targets
test:
	pytest tests/

test-cov:
	pytest tests/ --cov=coffee_law_verifier --cov-report=html --cov-report=term

# Code quality targets
lint:
	flake8 coffee_law_verifier tests

format:
	black coffee_law_verifier tests
	isort coffee_law_verifier tests

type-check:
	mypy coffee_law_verifier

# Clean targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .mypy_cache/ .pytest_cache/

# Documentation targets
docs:
	cd docs && sphinx-build -b html . _build/html

serve-docs:
	cd docs/_build/html && python -m http.server

# Running targets
run:
	cd coffee_law_verifier && python run_verification.py

run-openai:
	cd coffee_law_verifier && python run_verification.py --use-openai

# Benchmark target
benchmark:
	cd coffee_law_verifier && python -m cProfile -o profile.stats run_verification.py --samples 100
	@echo "Profile saved to coffee_law_verifier/profile.stats"
	@echo "View with: python -m pstats coffee_law_verifier/profile.stats"

# Development workflow targets
.PHONY: check ready

check: lint type-check test
	@echo "All checks passed!"

ready: format check
	@echo "Code is ready for commit!"