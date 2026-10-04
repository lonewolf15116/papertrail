.PHONY: install lint test eval up down

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check . && mypy

test:
	pytest

eval:
	papertrail-eval --gold data/gold/questions.jsonl

up:
	docker compose up --build -d

down:
	docker compose down
