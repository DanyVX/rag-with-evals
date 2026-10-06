.PHONY: setup lint format typecheck test test-fast bench run docker-build clean

setup:
	uv sync --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow and not gpu and not real_api"

bench:
	uv run ragx benchmark

run:
	uv run ragx serve

docker-build:
	@echo Docker support is intentionally deferred; use the local baseline.

clean:
	@echo Use git clean -fdX only after reviewing its target list.

