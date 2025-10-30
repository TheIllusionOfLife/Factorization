"""Shared test fixtures for all test files."""

from unittest.mock import AsyncMock

import pytest

from src.adaptive_engine import MetaLearningEngine
from src.comparison import BaselineStrategyGenerator
from src.config import Config
from src.crucible import FactorizationCrucible
from src.llm.base import LLMProvider, LLMResponse
from src.strategy import Strategy


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


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing without API calls."""
    provider = AsyncMock(spec=LLMProvider)
    provider.propose_mutation.return_value = LLMResponse(
        success=True,
        mutation_params={
            "mutation_type": "power",
            "parameters": {"new_power": 3},
        },
        reasoning="Test mutation for unit testing",
        cost=0.001,
        input_tokens=100,
        output_tokens=50,
        provider="test",
        model="test-model",
    )
    return provider


@pytest.fixture
def test_crucible():
    """Create a standard crucible for testing."""
    return FactorizationCrucible(961730063)


@pytest.fixture
def baseline_strategies():
    """Generate the three baseline strategies (conservative, balanced, aggressive)."""
    generator = BaselineStrategyGenerator()
    return generator.get_baseline_strategies()


@pytest.fixture
def sample_strategies():
    """Collection of test strategies for different scenarios."""
    return {
        "conservative": Strategy(
            power=2,
            modulus_filters=[(2, [0])],
            smoothness_bound=13,
            min_small_prime_hits=4,
        ),
        "balanced": Strategy(
            power=3,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=19,
            min_small_prime_hits=2,
        ),
        "aggressive": Strategy(
            power=4,
            modulus_filters=[(5, [0, 1, 2, 3, 4])],
            smoothness_bound=31,
            min_small_prime_hits=1,
        ),
    }


@pytest.fixture
def meta_learning_engine_with_history():
    """Create a meta-learning engine with 3 generations of history."""
    engine = MetaLearningEngine(adaptation_window=2)

    # Simulate 3 generations
    for gen in range(3):
        engine.current_generation = gen
        # Crossover: 5 offspring, 4 elite (80% success)
        for _i in range(5):
            engine.update_statistics("crossover", 50.0, _i < 4)
        # Mutation: 3 offspring, 1 elite (33% success)
        for _i in range(3):
            engine.update_statistics("mutation", 20.0, _i < 1)
        # Random: 2 offspring, 0 elite (0% success)
        for _i in range(2):
            engine.update_statistics("random", 10.0, False)
        engine.finalize_generation()

    return engine


@pytest.fixture
def timing_benchmark_config():
    """Config optimized for timing and performance tests."""
    return Config(
        api_key="test_key",
        llm_enabled=False,
        evaluation_duration=0.5,  # Longer for accurate timing measurements
        elite_selection_rate=0.2,
        crossover_rate=0.3,
        mutation_rate=0.5,
    )
