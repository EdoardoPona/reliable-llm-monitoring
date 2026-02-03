echo "Running formatting checks..."
uv run ruff format

echo "Running linting checks..."
uv run ruff check --fix

echo "Running type checks..."
uv run ty check reliable_monitoring experiments
