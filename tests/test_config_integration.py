"""Integration tests for config propagation through the system.

Tests verify that Config parameters correctly propagate to all components
and affect their behavior as expected.
"""

import pytest

from src.config import Config
from src.crucible import FactorizationCrucible
from src.evolution import EvolutionaryEngine
from src.strategy import StrategyGenerator


class TestConfigPropagationToGenerator:
    """Test that config propagates to StrategyGenerator and affects strategy generation."""

    def test_generator_respects_power_bounds(self):
        """Test that generated strategies respect config power bounds."""
        config = Config(api_key="", llm_enabled=False, power_min=3, power_max=3)
        generator = StrategyGenerator(config=config)

        # Generate multiple strategies
        strategies = [generator.random_strategy() for _ in range(10)]

        # All should have power=3 (min=max=3)
        for strategy in strategies:
            assert strategy.power == 3, (
                f"Strategy power {strategy.power} outside bounds [3, 3]"
            )

    def test_generator_respects_power_range(self):
        """Test that generated strategies stay within power range."""
        config = Config(api_key="", llm_enabled=False, power_min=2, power_max=4)
        generator = StrategyGenerator(config=config)

        # Generate many strategies to test randomness
        strategies = [generator.random_strategy() for _ in range(50)]

        for strategy in strategies:
            assert 2 <= strategy.power <= 4, (
                f"Strategy power {strategy.power} outside bounds [2, 4]"
            )

    def test_generator_respects_max_filters(self):
        """Test that generated strategies respect max_filters limit."""
        config = Config(api_key="", llm_enabled=False, max_filters=2)
        generator = StrategyGenerator(config=config)

        strategies = [generator.random_strategy() for _ in range(20)]

        for strategy in strategies:
            assert len(strategy.modulus_filters) <= 2, (
                f"Strategy has {len(strategy.modulus_filters)} filters, max is 2"
            )

    def test_generator_respects_min_hits_bounds(self):
        """Test that generated strategies respect min_hits bounds."""
        config = Config(api_key="", llm_enabled=False, min_hits_min=3, min_hits_max=5)
        generator = StrategyGenerator(config=config)

        strategies = [generator.random_strategy() for _ in range(30)]

        for strategy in strategies:
            assert 3 <= strategy.min_small_prime_hits <= 5, (
                f"Strategy min_hits {strategy.min_small_prime_hits} outside bounds [3, 5]"
            )


class TestConfigPropagationToEngine:
    """Test that config propagates through EvolutionaryEngine to all components."""

    def test_engine_passes_config_to_generator(self):
        """Test that engine creates generator with correct config."""
        crucible = FactorizationCrucible(961730063)
        config = Config(
            api_key="",
            llm_enabled=False,
            power_min=4,
            power_max=4,
            evaluation_duration=0.01,
        )

        # Engine creates its own generator
        engine = EvolutionaryEngine(
            crucible=crucible, population_size=5, config=config, random_seed=42
        )

        # Initialize population to generate strategies
        engine.initialize_population()

        # All civilizations should have power=4 (from config bounds)
        for civ_data in engine.civilizations.values():
            strategy = civ_data["strategy"]
            assert strategy.power == 4, (
                f"Engine-generated strategy has power {strategy.power}, expected 4"
            )

    def test_meta_learning_uses_config_fallback_rates(self):
        """Test that meta-learning engine receives and uses config fallback rates."""
        crucible = FactorizationCrucible(961730063)
        config = Config(
            api_key="",
            llm_enabled=False,
            evaluation_duration=0.01,
            fallback_inf_rate=0.9,  # Non-default value
            fallback_finite_rate=0.1,
        )

        engine = EvolutionaryEngine(
            crucible=crucible,
            population_size=5,
            config=config,
            enable_meta_learning=True,
            random_seed=42,
        )

        # Verify meta-learner was created with correct fallback rates
        assert engine.meta_learner is not None
        assert engine.meta_learner.fallback_inf_rate == 0.9
        assert engine.meta_learner.fallback_finite_rate == 0.1

    def test_meta_learning_uses_config_adaptation_window(self):
        """Test that meta-learning uses config adaptation window."""
        crucible = FactorizationCrucible(961730063)
        config = Config(
            api_key="",
            llm_enabled=False,
            evaluation_duration=0.01,
            adaptation_window=3,  # Non-default value
        )

        engine = EvolutionaryEngine(
            crucible=crucible,
            population_size=5,
            config=config,
            enable_meta_learning=True,
            random_seed=42,
        )

        assert engine.meta_learner is not None
        assert engine.meta_learner.adaptation_window == 3

    def test_meta_learning_uses_config_rate_bounds(self):
        """Test that meta-learning uses config min/max rate bounds."""
        crucible = FactorizationCrucible(961730063)
        config = Config(
            api_key="",
            llm_enabled=False,
            evaluation_duration=0.01,
            meta_learning_min_rate=0.15,  # Non-default
            meta_learning_max_rate=0.6,  # Non-default
        )

        engine = EvolutionaryEngine(
            crucible=crucible,
            population_size=5,
            config=config,
            enable_meta_learning=True,
            random_seed=42,
        )

        assert engine.meta_learner is not None
        assert engine.meta_learner.min_rate == 0.15
        assert engine.meta_learner.max_rate == 0.6


class TestConfigFromArgsAndEnv:
    """Test Config.from_args_and_env() factory method."""

    def test_factory_method_merges_cli_overrides(self):
        """Test that factory method correctly merges CLI args with defaults."""
        from argparse import Namespace

        args = Namespace(
            duration=0.05,  # Override
            elite_rate=0.25,  # Override
            crossover_rate=None,  # Use default
            mutation_rate=None,  # Use default
            power_min=None,
            power_max=None,
            max_filters=None,
            min_hits_min=None,
            min_hits_max=None,
            adaptation_window=None,
            meta_min_rate=None,
            meta_max_rate=None,
            fallback_inf_rate=None,
            fallback_finite_rate=None,
            mutation_prob_power=None,
            mutation_prob_filter=None,
            mutation_prob_modulus=None,
            mutation_prob_residue=None,
            mutation_prob_add_filter=None,
        )

        config = Config.from_args_and_env(args, use_llm=False)

        # Verify overrides applied
        assert config.evaluation_duration == 0.05
        assert config.elite_selection_rate == 0.25

        # Verify defaults used
        assert config.crossover_rate == 0.3  # Default
        assert config.mutation_rate == 0.5  # Default

    def test_factory_method_validates_combined_config(self):
        """Test that factory method validates combined config."""
        from argparse import Namespace

        args = Namespace(
            duration=None,
            elite_rate=None,
            crossover_rate=0.6,  # This + mutation_rate will exceed 1.0
            mutation_rate=0.5,
            power_min=None,
            power_max=None,
            max_filters=None,
            min_hits_min=None,
            min_hits_max=None,
            adaptation_window=None,
            meta_min_rate=None,
            meta_max_rate=None,
            fallback_inf_rate=None,
            fallback_finite_rate=None,
            mutation_prob_power=None,
            mutation_prob_filter=None,
            mutation_prob_modulus=None,
            mutation_prob_residue=None,
            mutation_prob_add_filter=None,
        )

        with pytest.raises(ValueError, match="Sum of crossover_rate"):
            Config.from_args_and_env(args, use_llm=False)


class TestEndToEndConfigFlow:
    """Test complete config flow from creation to strategy generation."""

    def test_full_evolution_cycle_respects_config_bounds(self):
        """Test that full evolution cycle produces strategies within config bounds."""
        crucible = FactorizationCrucible(961730063)
        config = Config(
            api_key="",
            llm_enabled=False,
            power_min=3,
            power_max=4,
            max_filters=2,
            min_hits_min=2,
            min_hits_max=4,
            evaluation_duration=0.01,
        )

        engine = EvolutionaryEngine(
            crucible=crucible, population_size=8, config=config, random_seed=42
        )

        engine.initialize_population()
        engine.run_evolutionary_cycle()

        # Check all civilizations respect bounds
        for civ_data in engine.civilizations.values():
            strategy = civ_data["strategy"]

            # Power bounds
            assert 3 <= strategy.power <= 4, (
                f"Strategy power {strategy.power} outside [3, 4]"
            )

            # Max filters
            assert len(strategy.modulus_filters) <= 2, (
                f"Strategy has {len(strategy.modulus_filters)} filters, max is 2"
            )

            # Min hits bounds
            assert 2 <= strategy.min_small_prime_hits <= 4, (
                f"Strategy min_hits {strategy.min_small_prime_hits} outside [2, 4]"
            )

    def test_mutation_respects_config_bounds(self):
        """Test that mutated strategies respect config bounds."""
        crucible = FactorizationCrucible(961730063)
        config = Config(
            api_key="",
            llm_enabled=False,
            power_min=2,
            power_max=3,
            max_filters=3,
            evaluation_duration=0.01,
        )

        engine = EvolutionaryEngine(
            crucible=crucible, population_size=10, config=config, random_seed=42
        )

        # Run multiple generations to trigger mutations
        engine.initialize_population()
        for _ in range(3):
            engine.run_evolutionary_cycle()

        # All strategies should still respect bounds
        for civ_data in engine.civilizations.values():
            strategy = civ_data["strategy"]
            assert 2 <= strategy.power <= 3
            assert len(strategy.modulus_filters) <= 3
