.PHONY: install lint test ingest eval up down

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check . && mypy

test:
	pytest

ingest:
	papertrail-ingest

eval:
	papertrail-eval --gold data/gold/questions.jsonl $(if $(wildcard data/processed/chunks.jsonl),--chunks data/processed/chunks.jsonl,)

up:
	docker compose up --build -d

down:
	docker compose down
