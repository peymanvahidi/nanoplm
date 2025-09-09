# nanoPLM Makefile for Testing and Coverage

# Variables
PYTHON := python
PIP := pip
VENV := .venv
VENV_ACTIVATE := source $(VENV)/bin/activate
COVERAGE_FILE := .coverage
COVERAGE_HTML_DIR := htmlcov
COVERAGE_XML := coverage.xml

# Default target
.PHONY: help
help:
	@echo "nanoPLM Test and Coverage Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  setup         - Set up virtual environment and install dependencies"
	@echo "  test          - Run all tests"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  test-fast     - Run tests in parallel for faster execution"
	@echo "  coverage      - Run tests with coverage report"
	@echo "  coverage-html - Generate HTML coverage report"
	@echo "  coverage-xml  - Generate XML coverage report for CI/CD"
	@echo "  coverage-all  - Generate both HTML and XML coverage reports"
	@echo "  coverage-open - Open HTML coverage report in browser"
	@echo "  clean         - Clean up coverage files and cache"
	@echo "  lint          - Run linting checks"
	@echo "  check-all     - Run tests, coverage, and linting"
	@echo ""

# Setup virtual environment and install dependencies
.PHONY: setup
setup:
	@echo "Setting up virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "Installing dependencies..."
	@$(VENV_ACTIVATE) && $(PIP) install -e .
	@$(VENV_ACTIVATE) && $(PIP) install -r tests/requirements.txt
	@echo "Setup complete!"

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	@$(VENV_ACTIVATE) && python -m pytest tests/

# Run tests with verbose output
.PHONY: test-verbose
test-verbose:
	@echo "Running tests with verbose output..."
	@$(VENV_ACTIVATE) && python -m pytest tests/ -v

# Run tests in parallel for faster execution
.PHONY: test-fast
test-fast:
	@echo "Running tests in parallel..."
	@$(VENV_ACTIVATE) && python -m pytest tests/ -n auto

# Run tests with coverage report
.PHONY: coverage
coverage:
	@echo "Running tests with coverage..."
	@$(VENV_ACTIVATE) && python -m pytest tests/ --cov=nanoplm --cov-report=term-missing

# Generate HTML coverage report
.PHONY: coverage-html
coverage-html: coverage
	@echo "Generating HTML coverage report..."
	@$(VENV_ACTIVATE) && python -m pytest tests/ --cov=nanoplm --cov-report=html
	@echo "HTML coverage report generated in $(COVERAGE_HTML_DIR)/"

# Generate XML coverage report for CI/CD
.PHONY: coverage-xml
coverage-xml: coverage
	@echo "Generating XML coverage report..."
	@$(VENV_ACTIVATE) && python -m pytest tests/ --cov=nanoplm --cov-report=xml
	@echo "XML coverage report generated as $(COVERAGE_XML)"

# Generate both HTML and XML coverage reports
.PHONY: coverage-all
coverage-all: coverage
	@echo "Generating HTML and XML coverage reports..."
	@$(VENV_ACTIVATE) && python -m pytest tests/ --cov=nanoplm --cov-report=html --cov-report=xml --cov-report=term-missing
	@echo "Coverage reports generated:"
	@echo "  - HTML: $(COVERAGE_HTML_DIR)/"
	@echo "  - XML: $(COVERAGE_XML)"

# Open HTML coverage report in browser (macOS)
.PHONY: coverage-open
coverage-open: coverage-html
	@echo "Opening HTML coverage report..."
	@if command -v open >/dev/null 2>&1; then \
		open $(COVERAGE_HTML_DIR)/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(COVERAGE_HTML_DIR)/index.html; \
	else \
		echo "Cannot open browser automatically. Open $(COVERAGE_HTML_DIR)/index.html manually."; \
	fi

# Clean up coverage files and cache
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf $(COVERAGE_HTML_DIR)
	@rm -f $(COVERAGE_FILE)
	@rm -f $(COVERAGE_XML)
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf nanoplm/__pycache__
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "Cleanup complete!"

# Run linting checks (if available)
.PHONY: lint
lint:
	@echo "Running linting checks..."
	@if command -v flake8 >/dev/null 2>&1; then \
		$(VENV_ACTIVATE) && flake8 nanoplm tests; \
	else \
		echo "Flake8 not found. Install with: pip install flake8"; \
	fi

# Comprehensive check: tests, coverage, and linting
.PHONY: check-all
check-all: test coverage-all lint
	@echo "All checks completed!"
	@echo "Check the coverage reports for detailed analysis."

# Show coverage summary
.PHONY: coverage-summary
coverage-summary:
	@if [ -f "$(COVERAGE_FILE)" ]; then \
		$(VENV_ACTIVATE) && python -m coverage report --show-missing; \
	else \
		echo "No coverage data found. Run 'make coverage' first."; \
	fi

# Run specific test file
.PHONY: test-file
test-file:
	@echo "Usage: make test-file TEST_FILE=path/to/test_file.py"
	@if [ -z "$(TEST_FILE)" ]; then \
		echo "Please specify TEST_FILE variable."; \
		echo "Example: make test-file TEST_FILE=tests/test_fasta_dataset.py"; \
		exit 1; \
	fi
	@echo "Running specific test file: $(TEST_FILE)"
	@$(VENV_ACTIVATE) && python -m pytest $(TEST_FILE) -v

# Run tests with coverage for specific file
.PHONY: coverage-file
coverage-file:
	@echo "Usage: make coverage-file TEST_FILE=path/to/test_file.py"
	@if [ -z "$(TEST_FILE)" ]; then \
		echo "Please specify TEST_FILE variable."; \
		echo "Example: make coverage-file TEST_FILE=tests/test_fasta_dataset.py"; \
		exit 1; \
	fi
	@echo "Running coverage for specific test file: $(TEST_FILE)"
	@$(VENV_ACTIVATE) && python -m pytest $(TEST_FILE) --cov=nanoplm --cov-report=term-missing -v

# Development setup with all dependencies
.PHONY: dev-setup
dev-setup: setup
	@echo "Installing development dependencies..."
	@$(VENV_ACTIVATE) && $(PIP) install black isort flake8 mypy
	@echo "Development environment ready!"

# Quick test run (fail fast)
.PHONY: test-quick
test-quick:
	@echo "Running quick test (fail fast)..."
	@$(VENV_ACTIVATE) && python -m pytest tests/ -x --tb=short

# Continuous integration target
.PHONY: ci
ci: setup test coverage-xml
	@echo "CI checks completed!"
