"""Shared test fixtures for all test files."""

import pytest

from src.config import Config


@pytest.fixture
def test_config():
    """Create a Config object with test-friendly defaults."""
    return Config(
        api_key="test_key",
        llm_enabled=False,
        evaluation_duration=0.05,  # Fast evaluation for tests
        elite_selection_rate=0.2,
        crossover_rate=0.3,
        mutation_rate=0.5,
    )


@pytest.fixture
def fast_test_config():
    """Create a Config object optimized for very fast tests."""
    return Config(
        api_key="test_key",
        llm_enabled=False,
        evaluation_duration=0.01,  # Ultra-fast evaluation
        elite_selection_rate=0.2,
        crossover_rate=0.3,
        mutation_rate=0.5,
    )
