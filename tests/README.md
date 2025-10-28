# nanoPLM Test Suite

This directory contains comprehensive tests for the nanoPLM project, focusing on the ModernBERT pretraining pipeline.

## Test Structure

- `test_data_collator.py` - Tests for data collation and padding functionality
- `test_fasta_dataset.py` - Tests for FASTA dataset creation and iteration
- `test_integration.py` - Integration tests for the full pipeline
- `test_smoke.py` - Basic smoke tests that can run without additional dependencies
- `conftest.py` - Pytest configuration and shared fixtures
- `test_runner.py` - Simple test runner for environments without pytest

## Running Tests

### Option 1: Simple Test Runner (No Dependencies)

```bash
# Run all tests
python tests/test_runner.py

# Run specific test files
python tests/test_runner.py --pattern "*collator*"

# Verbose output
python tests/test_runner.py --verbose

# List available test files
python tests/test_runner.py --list
```

### Option 2: Smoke Tests (Minimal Dependencies)

```bash
# Run basic functionality tests
python tests/test_smoke.py
```

### Option 3: Full Pytest Suite (Requires pytest)

First install test dependencies:
```bash
pip install -r tests/requirements.txt
```

Then run tests:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=nanoplm --cov-report=html

# Run specific test categories
pytest tests/ -m "integration" --run-integration
pytest tests/ -m "slow" --run-slow

# Run tests in parallel
pytest tests/ -n auto
```

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Smoke Tests**: Basic functionality verification

## Key Test Areas

### Data Collator Tests
- Verify padding works correctly for variable-length sequences
- Test MLM masking functionality
- Ensure attention masks are created properly

### FASTA Dataset Tests
- Test dataset creation from FASTA files
- Verify sequence tokenization and truncation
- Test memory efficiency and indexing

### Integration Tests
- Full pipeline from FASTA → Dataset → Collator → Model
- ModernBERT compatibility verification
- Memory usage and performance tests

## Test Data

Tests use sample FASTA files with protein sequences of varying lengths to ensure proper handling of:
- Different sequence lengths
- Special tokens (CLS, SEP, MASK)
- Padding and attention masks
- Truncation for long sequences

## Continuous Integration

For CI/CD pipelines, use:
```bash
# Quick smoke test
python tests/test_smoke.py

# Full test suite with coverage
pytest tests/ --cov=nanoplm --cov-report=xml --junitxml=results.xml
```

## Adding New Tests

1. Create test files following the naming pattern `test_*.py`
2. Use descriptive test method names starting with `test_`
3. Include docstrings explaining what each test verifies
4. Add appropriate fixtures in `conftest.py` if needed
5. Update this README if adding new test categories

## Test Fixtures

Shared fixtures available in `conftest.py`:
- `tokenizer`: ProtModernBertTokenizer instance
- `small_model`: Small model for testing (128 hidden size)
- `tiny_model`: Tiny model for fast testing (64 hidden size)
- `temp_fasta_file`: Temporary FASTA file with sample sequences
- `device`: Appropriate torch device (CPU/CUDA/MPS)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path or install package in development mode
2. **CUDA Out of Memory**: Use smaller models or reduce batch sizes in tests
3. **FASTA File Errors**: Check file permissions and format

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e .

# Install test dependencies
pip install -r tests/requirements.txt
```

## Performance Testing

For performance-sensitive tests:
- Use `tiny_model` fixture for fast iteration
- Skip slow tests in CI with `pytest -m "not slow"`
- Monitor memory usage in integration tests

## Coverage Goals

Target test coverage:
- Core functionality: >90%
- Error handling: >80%
- Edge cases: >70%
- Integration paths: >85%



