echo "Running formatting checks..."
ruff format

echo "Running linting checks..."
ruff check --fix

echo "Running type checks..."
uv run ty check
