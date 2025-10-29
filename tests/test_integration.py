"""Integration tests for LLM-enabled evolutionary engine"""

import pytest

from src.config import Config
from src.llm.base import LLMResponse
from src.llm.gemini import GeminiProvider


def test_llm_strategy_generator_init():
    """Test LLMStrategyGenerator initialization with and without provider"""
    from prototype import LLMStrategyGenerator

    # Without LLM provider
    gen = LLMStrategyGenerator()
    assert gen.llm_provider is None
    assert gen.fitness_history == []

    # With mock LLM provider
    config = Config(api_key="test")
    provider = GeminiProvider("test", config)
    gen_with_llm = LLMStrategyGenerator(llm_provider=provider)
    assert gen_with_llm.llm_provider is provider


def test_llm_strategy_generator_fallback_to_rule_based():
    """Test that LLMStrategyGenerator falls back to rule-based when LLM fails"""
    from prototype import LLMStrategyGenerator, Strategy

    # Create generator without LLM
    gen = LLMStrategyGenerator(llm_provider=None)

    parent = Strategy(
        power=2,
        modulus_filters=[(3, [0, 1])],
        smoothness_bound=13,
        min_small_prime_hits=2,
    )

    # Run multiple times to ensure at least one mutation happens (random may sometimes return same values)
    mutations = [
        gen.mutate_strategy_with_context(parent, fitness=50, generation=0)
        for _ in range(10)
    ]

    # Should get valid mutated strategies
    for child in mutations:
        assert isinstance(child, Strategy)
        assert 2 <= child.power <= 5
        assert len(child.modulus_filters) <= 4

    # At least one should be different from parent (random mutation)
    assert any(m != parent for m in mutations)


def test_llm_strategy_generator_uses_llm_on_success():
    """Test that LLMStrategyGenerator uses LLM when available and successful"""
    from unittest.mock import MagicMock

    from prototype import LLMStrategyGenerator, Strategy

    # Create mock LLM provider that returns success
    mock_provider = MagicMock()
    mock_provider.propose_mutation.return_value = LLMResponse(
        success=True,
        mutation_params={"mutation_type": "power", "parameters": {"new_power": 3}},
        reasoning="Test reasoning",
    )

    gen = LLMStrategyGenerator(llm_provider=mock_provider)
    parent = Strategy(
        power=2,
        modulus_filters=[(3, [0, 1])],
        smoothness_bound=13,
        min_small_prime_hits=2,
    )

    child = gen.mutate_strategy_with_context(parent, fitness=50, generation=5)

    # Should have called LLM
    mock_provider.propose_mutation.assert_called_once()
    # Should have new power
    assert child.power == 3


def test_llm_strategy_generator_fallback_on_llm_failure():
    """Test fallback to rule-based when LLM returns failure"""
    from unittest.mock import MagicMock

    from prototype import LLMStrategyGenerator, Strategy

    # Create mock LLM provider that returns failure
    mock_provider = MagicMock()
    mock_provider.propose_mutation.return_value = LLMResponse(
        success=False, mutation_params={}, error="API error"
    )

    gen = LLMStrategyGenerator(llm_provider=mock_provider)
    parent = Strategy(
        power=2,
        modulus_filters=[(3, [0, 1])],
        smoothness_bound=13,
        min_small_prime_hits=2,
    )

    # Run multiple times to ensure at least one mutation happens (random may sometimes return same values)
    mutations = [
        gen.mutate_strategy_with_context(parent, fitness=50, generation=5)
        for _ in range(10)
    ]

    # Should have called LLM multiple times
    assert mock_provider.propose_mutation.call_count == 10
    # Should fall back to rule-based
    for child in mutations:
        assert isinstance(child, Strategy)
    # At least one should be different from parent
    assert any(m != parent for m in mutations)


def test_evolutionary_engine_with_llm_provider():
    """Test EvolutionaryEngine accepts and uses LLM provider"""
    from prototype import EvolutionaryEngine, FactorizationCrucible
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test", max_llm_calls=0)  # No calls allowed
    provider = GeminiProvider("test", config)

    crucible = FactorizationCrucible(961730063)
    engine = EvolutionaryEngine(crucible, population_size=5, llm_provider=provider)

    # Should have LLM-enabled generator
    assert engine.generator.llm_provider is provider


def test_evolutionary_engine_without_llm_provider():
    """Test EvolutionaryEngine works without LLM (rule-based mode)"""
    from prototype import EvolutionaryEngine, FactorizationCrucible

    crucible = FactorizationCrucible(961730063)
    engine = EvolutionaryEngine(crucible, population_size=5)

    # Should have standard generator (no LLM)
    assert engine.generator.llm_provider is None


def test_evolutionary_engine_evaluation_duration():
    """Test EvolutionaryEngine uses configurable evaluation duration"""
    from prototype import EvolutionaryEngine, FactorizationCrucible

    crucible = FactorizationCrucible(961730063)
    engine = EvolutionaryEngine(
        crucible,
        population_size=3,
        config=Config(api_key="", llm_enabled=False, evaluation_duration=0.05),
    )

    assert engine.evaluation_duration == 0.05

    # Initialize population
    engine.initialize_population()

    # Run one cycle - this evaluates current gen and creates next gen
    _best_fitness, _best_strategy = engine.run_evolutionary_cycle()

    # After run_evolutionary_cycle, civilizations contains NEXT generation (fitness=0)
    # But we can verify the cycle completed quickly (thanks to short duration)
    # The important thing is that no exception was raised and duration was used
    assert engine.generation == 1  # Should have advanced to next generation


def test_apply_llm_mutation_power():
    """Test applying power mutation from LLM"""
    from prototype import LLMStrategyGenerator, Strategy

    gen = LLMStrategyGenerator()
    parent = Strategy(
        power=2,
        modulus_filters=[(3, [0, 1])],
        smoothness_bound=13,
        min_small_prime_hits=2,
    )

    mutation_params = {"mutation_type": "power", "parameters": {"new_power": 4}}

    child = gen._apply_llm_mutation(parent, mutation_params)
    assert child.power == 4
    assert child.modulus_filters == parent.modulus_filters


def test_apply_llm_mutation_add_filter():
    """Test applying add_filter mutation from LLM"""
    from prototype import LLMStrategyGenerator, Strategy

    gen = LLMStrategyGenerator()
    parent = Strategy(
        power=2,
        modulus_filters=[(3, [0, 1])],
        smoothness_bound=13,
        min_small_prime_hits=2,
    )

    mutation_params = {
        "mutation_type": "add_filter",
        "parameters": {"modulus": 5, "residues": [0, 1, 4]},
    }

    child = gen._apply_llm_mutation(parent, mutation_params)
    assert len(child.modulus_filters) == 2
    assert (5, [0, 1, 4]) in child.modulus_filters


def test_apply_llm_mutation_adjust_smoothness():
    """Test applying adjust_smoothness mutation from LLM"""
    from prototype import LLMStrategyGenerator, Strategy

    gen = LLMStrategyGenerator()
    parent = Strategy(
        power=2,
        modulus_filters=[(3, [0, 1])],
        smoothness_bound=13,
        min_small_prime_hits=2,
    )

    mutation_params = {
        "mutation_type": "adjust_smoothness",
        "parameters": {"bound_delta": 1, "hits_delta": -1},
    }

    child = gen._apply_llm_mutation(parent, mutation_params)
    assert child.smoothness_bound == 14  # 13 + 1
    assert child.min_small_prime_hits == 1  # 2 - 1


def test_first_generation_mutation_with_empty_history():
    """Test LLM mutation works with no prior fitness history"""
    from unittest.mock import MagicMock

    from prototype import LLMStrategyGenerator, Strategy

    # Create mock LLM provider that returns success
    mock_provider = MagicMock()
    mock_provider.propose_mutation.return_value = LLMResponse(
        success=True,
        mutation_params={"mutation_type": "power", "parameters": {"new_power": 3}},
        reasoning="Test reasoning for first generation",
    )

    gen = LLMStrategyGenerator(llm_provider=mock_provider)
    parent = Strategy(
        power=2,
        modulus_filters=[(3, [0, 1])],
        smoothness_bound=13,
        min_small_prime_hits=2,
    )

    # First generation with empty fitness history
    child = gen.mutate_strategy_with_context(parent, fitness=10, generation=0)

    # Should have called LLM with empty fitness history
    mock_provider.propose_mutation.assert_called_once()
    call_args = mock_provider.propose_mutation.call_args
    assert call_args[1]["fitness_history"] == []  # Empty list for first generation
    assert call_args[1]["generation"] == 0

    # Should have applied mutation
    assert child.power == 3


def test_llm_strategy_generator_empty_primes_validation():
    """Test that LLMStrategyGenerator rejects empty primes list"""
    from prototype import LLMStrategyGenerator

    with pytest.raises(ValueError, match="primes list cannot be empty"):
        LLMStrategyGenerator(llm_provider=None, primes=[])
