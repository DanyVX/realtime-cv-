.PHONY: setup lint format typecheck test test-fast bench run docker-build clean

setup:
	uv sync --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow and not gpu"

bench:
	uv run python -m rcs.bench.environment

run:
	uv run uvicorn rcs.server.app:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t realtime-cv-serving:local .

clean:
	uv cache clean
