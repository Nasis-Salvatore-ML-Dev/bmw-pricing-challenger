.PHONY: help install lint format test clean

help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies (dev)"
	@echo "  make lint       Run linting (flake8, pylint)"
	@echo "  make format     Format code (black, isort)"
	@echo "  make test       Run tests (pytest)"
	@echo "  make clean      Remove cache and build artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	flake8 src/ scripts/ tests/
	pylint src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov


install-gcp:
	pip install --upgrade pip &&\
		pip install -r requirements-gcp.txt



