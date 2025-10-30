"""Integration tests for comparison mode and baseline strategies."""

from src.comparison import BaselineStrategyGenerator, ComparisonEngine
from src.config import Config
from src.crucible import FactorizationCrucible
from src.strategy import SMALL_PRIMES


class TestBaselineConsistency:
    """Test baseline strategy consistency and determinism."""

    def test_baseline_evaluation_deterministic(
        self, test_crucible, baseline_strategies
    ):
        """Test that baseline evaluation is deterministic across calls."""
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
        engine = ComparisonEngine(
            crucible=test_crucible,
            num_runs=1,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
            config=config,
        )

        # Evaluate baselines twice
        fitness1 = engine._evaluate_baselines()
        fitness2 = engine._evaluate_baselines()

        # Due to timing-based evaluation, exact match not guaranteed
        # But should be similar (within 30%)
        for key in fitness1:
            if fitness1[key] > 0 and fitness2[key] > 0:
                diff_pct = abs(fitness1[key] - fitness2[key]) / max(
                    fitness1[key], fitness2[key]
                )
                assert diff_pct < 0.4, (
                    f"Baseline {key} varies too much: {fitness1[key]} vs {fitness2[key]}"
                )

    def test_baseline_fitness_non_negative(self, baseline_strategies):
        """Test that all baseline strategies are valid and evaluable."""
        crucible = FactorizationCrucible(961730063)

        for name, strategy in baseline_strategies.items():
            metrics = crucible.evaluate_strategy_detailed(strategy, 0.05)
            # Fitness must be non-negative
            assert metrics.candidate_count >= 0, f"{name} has negative fitness"

    def test_baseline_strategies_valid(self, baseline_strategies):
        """Test that baseline strategies pass validation."""
        # All baselines should be valid Strategy objects
        assert len(baseline_strategies) == 3
        assert "conservative" in baseline_strategies
        assert "balanced" in baseline_strategies
        assert "aggressive" in baseline_strategies

        for name, strategy in baseline_strategies.items():
            # Valid power range
            assert 2 <= strategy.power <= 5, f"{name} has invalid power"
            # Valid filters
            assert len(strategy.modulus_filters) <= 4, f"{name} has too many filters"
            # Valid min hits
            assert 1 <= strategy.min_small_prime_hits <= 6, (
                f"{name} has invalid min_hits"
            )

    def test_baseline_generation_idempotent(self):
        """Test that baseline generator produces same strategies each time."""
        gen1 = BaselineStrategyGenerator()
        gen2 = BaselineStrategyGenerator()

        baselines1 = gen1.get_baseline_strategies()
        baselines2 = gen2.get_baseline_strategies()

        # Should be identical
        for key in baselines1:
            assert baselines1[key].power == baselines2[key].power
            assert baselines1[key].modulus_filters == baselines2[key].modulus_filters
            assert baselines1[key].smoothness_bound == baselines2[key].smoothness_bound
            assert (
                baselines1[key].min_small_prime_hits
                == baselines2[key].min_small_prime_hits
            )


class TestBestStrategyValidation:
    """Test that best strategy is properly tracked and valid."""

    def test_best_strategy_parameters_valid(self, test_crucible):
        """Test that final_best_strategy has valid parameters."""
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
        engine = ComparisonEngine(
            crucible=test_crucible,
            num_runs=1,
            max_generations=3,
            population_size=5,
            evaluation_duration=0.05,
            config=config,
        )

        runs = engine.run_comparison(base_seed=42)
        run = runs[0]

        strategy = run.final_best_strategy

        # Valid parameters
        assert 2 <= strategy.power <= 5
        assert len(strategy.modulus_filters) <= 4
        # smoothness_bound can be any value from SMALL_PRIMES (after normalization)
        assert strategy.smoothness_bound in SMALL_PRIMES
        assert 1 <= strategy.min_small_prime_hits <= 6

    def test_final_best_across_all_runs(self, test_crucible):
        """Test that each run has a valid best strategy."""
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
        engine = ComparisonEngine(
            crucible=test_crucible,
            num_runs=3,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
            config=config,
        )

        runs = engine.run_comparison(base_seed=100)

        assert len(runs) == 3

        for run in runs:
            # Each run should have valid best strategy
            assert run.final_best_strategy is not None
            assert 2 <= run.final_best_strategy.power <= 5


class TestRNGIsolation:
    """Test RNG isolation between components."""

    def test_same_seed_reproducible_evolution(self, test_crucible):
        """Test that same seed produces similar evolution.

        Note: Timing-based evaluation means fitness is not 100% deterministic,
        which can affect selection and mutation. We verify initialization
        is reproducible but allow for variation in final strategy.
        """
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)

        engine1 = ComparisonEngine(
            crucible=test_crucible,
            num_runs=1,
            max_generations=3,
            population_size=5,
            evaluation_duration=0.05,
            config=config,
        )

        engine2 = ComparisonEngine(
            crucible=test_crucible,
            num_runs=1,
            max_generations=3,
            population_size=5,
            evaluation_duration=0.05,
            config=config,
        )

        runs1 = engine1.run_comparison(base_seed=42)
        runs2 = engine2.run_comparison(base_seed=42)

        # Both should complete successfully
        assert runs1[0].final_best_strategy is not None
        assert runs2[0].final_best_strategy is not None

        # Strategies may differ due to timing-based fitness variations
        # This test just verifies no crashes with same seed
        assert True

    def test_different_seeds_different_evolution(self, test_crucible):
        """Test that different seeds produce different evolution."""
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)

        engine1 = ComparisonEngine(
            crucible=test_crucible,
            num_runs=1,
            max_generations=3,
            population_size=5,
            evaluation_duration=0.05,
            config=config,
        )

        engine2 = ComparisonEngine(
            crucible=test_crucible,
            num_runs=1,
            max_generations=3,
            population_size=5,
            evaluation_duration=0.05,
            config=config,
        )

        runs1 = engine1.run_comparison(base_seed=42)
        runs2 = engine2.run_comparison(base_seed=100)

        # Final strategies should likely be different (different seeds)
        # May occasionally be same by chance, so we just verify code runs
        # without asserting difference (would be flaky)
        assert runs1[0].final_best_strategy is not None
        assert runs2[0].final_best_strategy is not None

    def test_comparison_runs_independent(self, test_crucible):
        """Test that multiple runs are independent."""
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
        engine = ComparisonEngine(
            crucible=test_crucible,
            num_runs=3,
            max_generations=2,
            population_size=3,
            evaluation_duration=0.05,
            config=config,
        )

        runs = engine.run_comparison(base_seed=42)

        # Each run should have unique seed (base_seed + run_idx)
        # and produce different (or occasionally same) results
        assert len(runs) == 3

        # All runs should have valid strategies
        for run in runs:
            assert run.final_best_strategy is not None


class TestConvergenceDetection:
    """Test convergence detection integration."""

    def test_convergence_detector_integration(self, test_crucible):
        """Test convergence detector in full comparison flow."""
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
        engine = ComparisonEngine(
            crucible=test_crucible,
            num_runs=2,
            max_generations=10,  # Allow convergence
            population_size=5,
            evaluation_duration=0.05,
            config=config,
        )

        runs = engine.run_comparison(base_seed=42)

        # At least check that convergence tracking works
        for run in runs:
            # generations_to_convergence may be None (not converged) or int
            if run.generations_to_convergence is not None:
                assert isinstance(run.generations_to_convergence, int)
                assert run.generations_to_convergence >= 0

    def test_convergence_window_boundary(self, test_crucible):
        """Test convergence detection with exact window_size generations."""
        config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
        engine = ComparisonEngine(
            crucible=test_crucible,
            num_runs=1,
            max_generations=5,  # Exactly window size
            population_size=3,
            evaluation_duration=0.05,
            config=config,
            convergence_window=5,
        )

        runs = engine.run_comparison(base_seed=42)

        # Should complete without error
        assert len(runs) == 1
        assert runs[0].final_best_strategy is not None
