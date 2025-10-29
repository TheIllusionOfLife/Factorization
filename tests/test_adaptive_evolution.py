"""
Integration tests for meta-learning with EvolutionaryEngine.

Tests cover:
- Engine initialization with meta-learning
- Operator metadata tracking
- Rate adaptation during evolution
- Export of operator history
"""

import json
import random
import tempfile
from pathlib import Path

from prototype import (
    EvolutionaryEngine,
    FactorizationCrucible,
    StrategyGenerator,
)
from src.meta_learning import OperatorMetadata


class TestMetaLearningEngineIntegration:
    """Test meta-learning integration with EvolutionaryEngine."""

    def test_engine_initialization_without_meta_learning(self):
        """Test engine initializes without meta-learning by default."""
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=5,
            evaluation_duration=0.01,
        )
        assert engine.meta_learner is None
        assert not hasattr(engine, "rate_history") or len(engine.rate_history) == 0

    def test_engine_initialization_with_meta_learning(self):
        """Test engine initializes with meta-learning enabled."""
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=5,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            adaptation_window=3,
        )
        assert engine.meta_learner is not None
        assert engine.meta_learner.adaptation_window == 3
        assert hasattr(engine, "rate_history")

    def test_civilizations_have_operator_metadata(self):
        """Test that all civilizations have operator metadata after generation."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=8,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        # Run one generation
        best_fitness, best_strategy = engine.run_evolutionary_cycle()

        # All civilizations should have operator metadata
        for _civ_id, civ_data in engine.civilizations.items():
            assert "operator_metadata" in civ_data
            metadata = civ_data["operator_metadata"]
            assert isinstance(metadata, OperatorMetadata)
            assert metadata.operator in ["crossover", "mutation", "random"]
            assert metadata.generation == 1

    def test_initial_generation_all_random(self):
        """Test that generation 0 has all random operators."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=5,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            random_seed=42,
        )

        # Initialize population
        engine.initialize_population()

        # Check initial generation
        for civ_data in engine.civilizations.values():
            assert civ_data["operator_metadata"].operator == "random"
            assert civ_data["operator_metadata"].generation == 0

    def test_operators_tracked_across_generations(self):
        """Test that operators are tracked correctly across multiple generations."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=10,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        # Run 3 generations
        for _gen in range(3):
            engine.run_evolutionary_cycle()

        # Check that we have history for 2 generations
        # Generation 0 is skipped (condition: self.generation > 0)
        # run_evolutionary_cycle() #1: generation=0, skips statistics
        # run_evolutionary_cycle() #2: generation=1, collects & finalizes stats
        # run_evolutionary_cycle() #3: generation=2, collects & finalizes stats
        assert len(engine.meta_learner.operator_history) == 2

        # Each generation should have stats for all three operators
        for gen_stats in engine.meta_learner.operator_history:
            assert "crossover" in gen_stats
            assert "mutation" in gen_stats
            assert "random" in gen_stats

    def test_rates_adapt_after_window(self):
        """Test that rates change after adaptation window."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=10,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            adaptation_window=3,
            random_seed=42,
        )

        # Store initial rates
        initial_crossover = engine.crossover_rate
        initial_mutation = engine.mutation_rate
        initial_random = 1.0 - initial_crossover - initial_mutation

        # Initialize population before running
        engine.initialize_population()

        # Run past adaptation window
        for _ in range(5):
            engine.run_evolutionary_cycle()

        # Rates should have changed
        final_crossover = engine.crossover_rate
        final_mutation = engine.mutation_rate
        final_random = 1.0 - final_crossover - final_mutation

        # At least one rate should have changed (>2% to allow for conservative bounds)
        rate_changed = (
            abs(final_crossover - initial_crossover) > 0.02
            or abs(final_mutation - initial_mutation) > 0.02
            or abs(final_random - initial_random) > 0.02
        )
        assert rate_changed, (
            f"Rates should adapt after window. "
            f"Initial: c={initial_crossover:.3f}, m={initial_mutation:.3f}, r={initial_random:.3f}. "
            f"Final: c={final_crossover:.3f}, m={final_mutation:.3f}, r={final_random:.3f}"
        )

    def test_rates_sum_to_one_after_adaptation(self):
        """Test that adapted rates always sum to 1.0."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=10,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            adaptation_window=2,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        # Run multiple generations
        for _ in range(6):
            engine.run_evolutionary_cycle()

            # Check rates sum to 1.0 after each generation
            total_rate = (
                engine.crossover_rate + engine.mutation_rate + engine.random_rate
            )
            assert abs(total_rate - 1.0) < 1e-6

    def test_operator_history_exported(self):
        """Test that operator history is exported to JSON."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=8,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            adaptation_window=2,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        # Run a few generations
        for _ in range(4):
            engine.run_evolutionary_cycle()

        # Export to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            engine.export_metrics(temp_path)

            # Read and verify JSON
            with open(temp_path) as f:
                data = json.load(f)

            assert "operator_history" in data
            assert data["operator_history"] is not None
            # Generation 0 skips statistics, so 4 cycles = 3 history entries
            assert len(data["operator_history"]) == 3

            # Check structure of operator history
            for gen_data in data["operator_history"]:
                assert "generation" in gen_data
                assert "rates" in gen_data
                assert "operator_stats" in gen_data
                assert "crossover" in gen_data["operator_stats"]
                assert "mutation" in gen_data["operator_stats"]
                assert "random" in gen_data["operator_stats"]

        finally:
            Path(temp_path).unlink()

    def test_operator_history_not_exported_without_meta_learning(self):
        """Test that operator history is None when meta-learning disabled."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=5,
            evaluation_duration=0.01,
            enable_meta_learning=False,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        engine.run_evolutionary_cycle()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            engine.export_metrics(temp_path)

            with open(temp_path) as f:
                data = json.load(f)

            assert "operator_history" in data
            assert data["operator_history"] is None

        finally:
            Path(temp_path).unlink()

    def test_parent_fitness_tracked_correctly(self):
        """Test that parent fitness is tracked in operator metadata."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=10,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        # Run one generation to create parents
        engine.run_evolutionary_cycle()

        # Run second generation
        engine.run_evolutionary_cycle()

        # Check that non-random operators have parent fitness
        for _civ_id, civ_data in engine.civilizations.items():
            metadata = civ_data["operator_metadata"]
            if metadata.operator in ["crossover", "mutation"]:
                assert len(metadata.parent_fitness) > 0
                # Parent fitness should be non-negative
                for pf in metadata.parent_fitness:
                    assert pf >= 0

    def test_meta_learning_with_small_population(self):
        """Test meta-learning works with small population (edge case)."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=3,  # Very small
            evaluation_duration=0.01,
            enable_meta_learning=True,
            adaptation_window=2,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        # Should run without errors
        for _ in range(3):
            best_fitness, best_strategy = engine.run_evolutionary_cycle()
            assert best_fitness >= 0

    def test_rate_history_tracking(self):
        """Test that rate history is tracked over generations."""
        random.seed(42)
        crucible = FactorizationCrucible(961730063)
        generator = StrategyGenerator()
        engine = EvolutionaryEngine(
            crucible=crucible,
            generator=generator,
            population_size=10,
            evaluation_duration=0.01,
            enable_meta_learning=True,
            adaptation_window=2,
            random_seed=42,
        )

        # Initialize population before running
        engine.initialize_population()

        # Run several generations
        for _ in range(5):
            engine.run_evolutionary_cycle()

        # Check rate history
        # Should have 6 entries: 1 from initialize_population + 5 from cycles
        assert hasattr(engine, "rate_history")
        assert len(engine.rate_history) == 6

        # Each entry should have all three rates
        for rates in engine.rate_history:
            assert "crossover" in rates
            assert "mutation" in rates
            assert "random" in rates
            total = rates["crossover"] + rates["mutation"] + rates["random"]
            assert abs(total - 1.0) < 1e-6
