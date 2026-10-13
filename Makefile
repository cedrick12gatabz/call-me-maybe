run:
	uv run python -m src

install:
	uv pip install -r pyproject.toml

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	uv run flake8 src/ --exclude=.venv
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/ --exclude=.env
	uv run mypy src/ --strict
