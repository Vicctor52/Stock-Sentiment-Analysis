.PHONY: sync sync-dashboard lint format test ingest score analyze dashboard

sync:
	uv venv --python 3.12
	uv pip install -r requirements-pipeline.txt

sync-dashboard:
	uv venv .venv-dashboard --python 3.12
	uv pip install --python .venv-dashboard -r requirements.txt

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

ingest:
	uv run python -m finsent ingest

score:
	uv run python -m finsent score

analyze:
	uv run python -m finsent analyze

dashboard:
	uv run streamlit run streamlit_app.py
