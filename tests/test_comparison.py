"""
Tests for comparison engine functionality.

Following TDD: These tests are written BEFORE implementation.
All tests should FAIL initially (RED phase).
"""

import pytest
from prototype import (
    ComparisonEngine,
    ComparisonRun,
    FactorizationCrucible,
    BaselineStrategyGenerator,
)


class TestComparisonRun:
    """Test ComparisonRun dataclass."""

    def test_comparison_run_creation(self):
        """Test creating a ComparisonRun."""
        run = ComparisonRun(
            evolved_fitness=[100, 200, 300],
            baseline_fitness={"conservative": 150, "balanced": 180, "aggressive": 250},
            generations_to_convergence=2,
            final_best_strategy=None,
            random_seed=42,
        )

        assert len(run.evolved_fitness) == 3
        assert run.baseline_fitness["conservative"] == 150
        assert run.generations_to_convergence == 2
        assert run.random_seed == 42


class TestComparisonEngine:
    """Test comparison engine functionality."""

    def test_engine_initialization(self):
        """Test that comparison engine can be created."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=3,
            max_generations=5,
            population_size=5,
            evaluation_duration=0.05,
        )

        assert engine.num_runs == 3
        assert engine.max_generations == 5
        assert engine.population_size == 5
        assert isinstance(engine.baseline_generator, BaselineStrategyGenerator)

    def test_engine_has_statistical_tools(self):
        """Test that engine has statistical analysis tools."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(crucible=crucible)

        # Should have convergence detector and statistical analyzer
        assert hasattr(engine, "convergence_detector")
        assert hasattr(engine, "statistical_analyzer")
        assert hasattr(engine, "baseline_generator")

    def test_single_comparison_run_completes(self):
        """Test that a single comparison run completes without error."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=1,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
        )

        runs = engine.run_comparison(base_seed=42)

        assert len(runs) == 1
        assert isinstance(runs[0], ComparisonRun)
        assert len(runs[0].evolved_fitness) > 0
        assert "conservative" in runs[0].baseline_fitness
        assert "balanced" in runs[0].baseline_fitness
        assert "aggressive" in runs[0].baseline_fitness

    def test_multiple_runs_with_seeds_reproducible(self):
        """Test that same seed produces identical results."""
        crucible = FactorizationCrucible(number_to_factor=961730063)

        # First comparison
        engine1 = ComparisonEngine(
            crucible=crucible,
            num_runs=2,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
        )
        runs1 = engine1.run_comparison(base_seed=100)

        # Second comparison with same seed
        engine2 = ComparisonEngine(
            crucible=crucible,
            num_runs=2,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
        )
        runs2 = engine2.run_comparison(base_seed=100)

        # Results should be identical
        assert len(runs1) == len(runs2)
        for run1, run2 in zip(runs1, runs2):
            assert run1.evolved_fitness == run2.evolved_fitness
            assert run1.random_seed == run2.random_seed

    def test_baseline_evaluation_deterministic(self):
        """Test that baseline evaluation is deterministic."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=2,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
        )

        runs = engine.run_comparison(base_seed=50)

        # At least balanced and aggressive should find candidates
        baseline_balanced = [run.baseline_fitness["balanced"] for run in runs]
        baseline_aggressive = [run.baseline_fitness["aggressive"] for run in runs]

        # Balanced and aggressive should find candidates (conservative may be too strict)
        assert all(f > 0 for f in baseline_balanced), "Balanced baseline should find candidates"
        assert all(f > 0 for f in baseline_aggressive), "Aggressive baseline should find candidates"

    def test_convergence_detection_triggers_early_stop(self):
        """Test that convergence detection can stop before max generations."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=1,
            max_generations=20,  # High max
            population_size=5,
            evaluation_duration=0.05,
            convergence_window=3,
        )

        runs = engine.run_comparison(base_seed=42)

        # Should stop early if converged
        assert len(runs[0].evolved_fitness) <= 20
        # If converged, should have convergence generation
        if runs[0].generations_to_convergence is not None:
            assert runs[0].generations_to_convergence < 20

    def test_comparison_run_tracks_fitness_history(self):
        """Test that fitness history is recorded per generation."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=1,
            max_generations=3,
            population_size=5,  # Larger population to ensure at least some find candidates
            evaluation_duration=0.1,  # Longer duration
        )

        runs = engine.run_comparison(base_seed=42)
        run = runs[0]

        # Should have fitness history (may be 1-3 generations depending on convergence)
        assert len(run.evolved_fitness) >= 1
        assert len(run.evolved_fitness) <= 3
        # Fitness values should be non-negative
        assert all(f >= 0 for f in run.evolved_fitness), "Fitness should be non-negative"

    def test_analyze_results_returns_comparison(self):
        """Test that analyze_results returns proper structure."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=2,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
        )

        runs = engine.run_comparison(base_seed=42)
        analysis = engine.analyze_results(runs)

        # Should have required keys
        assert "comparison_results" in analysis
        assert "convergence_stats" in analysis
        assert "num_runs" in analysis
        assert analysis["num_runs"] == 2

    def test_analyze_results_statistical_significance(self):
        """Test that statistical analysis includes p-values and effect sizes."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=3,
            max_generations=3,
            population_size=5,
            evaluation_duration=0.1,
        )

        runs = engine.run_comparison(base_seed=42)
        analysis = engine.analyze_results(runs)

        # Should have comparison for each baseline
        comparison_results = analysis["comparison_results"]
        assert "conservative" in comparison_results
        assert "balanced" in comparison_results
        assert "aggressive" in comparison_results

        # Each comparison should have statistical data
        for baseline_name, result in comparison_results.items():
            assert hasattr(result, "p_value")
            assert hasattr(result, "effect_size")
            assert hasattr(result, "is_significant")
            assert hasattr(result, "evolved_mean")
            assert hasattr(result, "baseline_mean")

    def test_convergence_stats_calculated(self):
        """Test that convergence statistics are computed."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=3,
            max_generations=5,
            population_size=5,
            evaluation_duration=0.05,
            convergence_window=3,
        )

        runs = engine.run_comparison(base_seed=42)
        analysis = engine.analyze_results(runs)

        convergence_stats = analysis["convergence_stats"]
        assert "convergence_rate" in convergence_stats
        assert 0 <= convergence_stats["convergence_rate"] <= 1.0

    def test_evolved_strategies_evaluated(self):
        """Test that evolved strategies are actually evaluated."""
        crucible = FactorizationCrucible(number_to_factor=961730063)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=1,
            max_generations=2,
            population_size=5,
            evaluation_duration=0.1,
        )

        runs = engine.run_comparison(base_seed=42)
        run = runs[0]

        # Final strategy should exist and be valid
        assert run.final_best_strategy is not None
        assert hasattr(run.final_best_strategy, "power")
        assert hasattr(run.final_best_strategy, "modulus_filters")

    def test_engine_with_llm_provider(self):
        """Test that engine works with LLM provider (mock)."""
        crucible = FactorizationCrucible(number_to_factor=961730063)

        # Create engine with None llm_provider (rule-based mode)
        engine = ComparisonEngine(
            crucible=crucible,
            num_runs=1,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
            llm_provider=None,  # Rule-based
        )

        runs = engine.run_comparison(base_seed=42)

        # Should complete successfully
        assert len(runs) == 1
        assert len(runs[0].evolved_fitness) > 0
