# Makefile for Factorization Project
# Provides convenient commands for local CI validation and development

.PHONY: help ci ci-fast lint format test test-fast test-cov install-hooks clean

# Default target: show help
help:
	@echo "Factorization Project - Makefile Commands"
	@echo ""
	@echo "Local CI Validation:"
	@echo "  make ci           - Run all CI checks locally (lint + format + test + mypy)"
	@echo "  make ci-fast      - Run fast CI checks (lint + format + fast tests)"
	@echo ""
	@echo "Individual Checks:"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Run ruff formatter (auto-fix)"
	@echo "  make format-check - Check formatting without modifying files"
	@echo "  make test         - Run all tests with pytest"
	@echo "  make test-fast    - Run fast tests only (exclude integration)"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make type-check   - Run mypy type checking"
	@echo ""
	@echo "Pre-commit Hooks:"
	@echo "  make install-hooks - Install pre-commit hooks (run once after clone)"
	@echo "  make hooks         - Run all pre-commit hooks manually"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache and temporary files"

# Run all CI checks (matches GitHub Actions CI)
ci: lint format-check test type-check
	@echo "✅ All CI checks passed!"

# Run fast CI checks (for quick local validation)
ci-fast: lint format-check test-fast
	@echo "✅ Fast CI checks passed!"

# Ruff linter (matches CI: ruff check .)
lint:
	@echo "Running ruff linter..."
	ruff check .

# Ruff formatter - auto-fix (applies changes)
format:
	@echo "Running ruff formatter (auto-fix)..."
	ruff format .

# Ruff formatter - check only (matches CI: ruff format --check .)
format-check:
	@echo "Checking code formatting..."
	ruff format --check .

# Run all tests (matches CI: pytest tests/ -v --tb=short)
test:
	@echo "Running all tests..."
	pytest tests/ -v --tb=short

# Run fast tests only (exclude integration tests)
test-fast:
	@echo "Running fast tests (excluding integration)..."
	pytest tests/ -v --tb=short -m "not integration" --maxfail=3

# Run tests with coverage report
test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov=. --cov-report=term-missing --cov-report=html
	@echo "Coverage report: file://$(PWD)/htmlcov/index.html"

# Type checking with mypy (matches CI: mypy src/ --ignore-missing-imports)
type-check:
	@echo "Running mypy type checking..."
	mypy src/ --ignore-missing-imports --no-strict-optional

# Install pre-commit hooks (run once after cloning repository)
install-hooks:
	@echo "Installing pre-commit hooks..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "✅ Pre-commit hooks installed successfully!"; \
		echo "   Hooks will now run automatically on 'git commit'"; \
		echo "   To bypass: git commit --no-verify (emergency only)"; \
	else \
		echo "❌ Error: pre-commit not found"; \
		echo "   Install with: pip install -r requirements-dev.txt"; \
		exit 1; \
	fi

# Run all pre-commit hooks manually
hooks:
	@echo "Running pre-commit hooks on all files..."
	pre-commit run --all-files

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"
