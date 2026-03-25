#!/bin/bash
# scripts/test_local.sh -- Run the test suite locally
set -e

echo "Running unit tests..."
uv run pytest tests/ -x -v --tb=short

echo "All tests passed."
