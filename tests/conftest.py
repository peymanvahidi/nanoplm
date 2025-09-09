"""
Pytest configuration and shared fixtures for the nanoPLM test suite.
"""
import pytest
import tempfile
import os
import torch
from pathlib import Path

from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@pytest.fixture(scope="session")
def tokenizer():
    """Create a tokenizer instance for the test session."""
    return ProtModernBertTokenizer()


@pytest.fixture
def sample_fasta_content():
    """Sample FASTA content for testing."""
    return """>protein1
MKALCLLLLPVLGLLTGSSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGS
>protein2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLS
>protein3
MAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGT
"""


@pytest.fixture
def temp_fasta_file(sample_fasta_content):
    """Create a temporary FASTA file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(sample_fasta_content)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def small_model(tokenizer):
    """Create a small model for testing."""
    model = ProtModernBertMLM(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
        mlp_activation="swiglu",
        mlp_dropout=0.0,
        mlp_bias=False,
        attention_bias=False,
        attention_dropout=0.0,
        classifier_activation="gelu"
    )
    return model


@pytest.fixture
def tiny_model(tokenizer):
    """Create a tiny model for fast testing."""
    model = ProtModernBertMLM(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=tokenizer.vocab_size,
        mlp_activation="swiglu",
        mlp_dropout=0.0,
        mlp_bias=False,
        attention_bias=False,
        attention_dropout=0.0,
        classifier_activation="gelu"
    )
    return model


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and content."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark GPU tests
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark slow tests
        if "slow" in item.name.lower() or "memory" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Custom pytest options
def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_runtest_setup(item):
    """Setup for each test."""
    # Skip integration tests unless explicitly requested
    if item.get_closest_marker("integration"):
        if not item.config.getoption("--run-integration"):
            pytest.skip("Integration test skipped. Use --run-integration to run.")

    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow"):
        if not item.config.getoption("--run-slow"):
            pytest.skip("Slow test skipped. Use --run-slow to run.")
