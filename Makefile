.PHONY: help install dev-install test lint docs clean run-dev deploy-local backup

PYTHON := python
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
MYPY := $(VENV)/bin/mypy

help:
	@echo "Star Trek Technology Development Commands"
	@echo "----------------------------------------"
	@echo "make install      - Install production dependencies"
	@echo "make dev-install  - Install development dependencies"
	@echo "make test         - Run tests"
	@echo "make lint         - Run code quality checks"
	@echo "make docs         - Build documentation"
	@echo "make clean        - Clean build artifacts"
	@echo "make run-dev      - Run development server"
	@echo "make deploy-local - Deploy locally using Docker"
	@echo "make backup       - Create local backup"

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev-install: install
	$(PIP) install -r requirements-dev.txt
	pre-commit install

test:
	$(PYTEST) tests/ --cov=src/ --cov-report=html

lint:
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/
	$(MYPY) src/

docs:
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf docs/_build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-dev:
	$(PYTHON) src/app.py --config config/development.yaml

deploy-local:
	docker-compose up --build -d
	@echo "Local deployment running at http://localhost:8000"

backup:
	./scripts/backup.sh local

# Environment setup
setup-env:
	cp .env.example .env
	@echo "Please update .env with your configuration"

# Database operations
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# Monitoring
start-monitoring:
	docker-compose -f docker-compose.monitoring.yml up -d

# Development tools
jupyter:
	jupyter lab

format:
	$(BLACK) .
	$(ISORT) .

type-check:
	$(MYPY) .

security-check:
	bandit -r src/

# Documentation
serve-docs:
	cd docs/_build/html && python -m http.server 8080

# Container operations
build:
	docker build -t startrektech .

run:
	docker run -p 8000:8000 startrektech

# Kubernetes operations
k8s-deploy:
	kubectl apply -f kubernetes/

k8s-status:
	kubectl get all -n startrektech
