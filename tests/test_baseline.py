"""
Tests for baseline strategy generation.

Following TDD: These tests are written BEFORE implementation.
All tests should FAIL initially (RED phase).
"""

import pytest
from prototype import (
    BaselineStrategyGenerator,
    Strategy,
    FactorizationCrucible,
    SMALL_PRIMES,
)


class TestBaselineStrategyGenerator:
    """Test baseline strategy generation functionality."""

    def test_generator_initialization(self):
        """Test that baseline generator can be created."""
        generator = BaselineStrategyGenerator()
        assert generator.primes == SMALL_PRIMES

    def test_get_baseline_strategies_returns_dict(self):
        """Test that baseline strategies are returned as a dict."""
        generator = BaselineStrategyGenerator()
        baselines = generator.get_baseline_strategies()

        assert isinstance(baselines, dict)
        assert len(baselines) == 3
        assert "conservative" in baselines
        assert "balanced" in baselines
        assert "aggressive" in baselines

    def test_baseline_strategies_are_strategy_objects(self):
        """Test that all baseline strategies are Strategy instances."""
        generator = BaselineStrategyGenerator()
        baselines = generator.get_baseline_strategies()

        for name, strategy in baselines.items():
            assert isinstance(
                strategy, Strategy
            ), f"{name} should be a Strategy instance"

    def test_baseline_strategies_are_valid(self):
        """Test that all baseline strategies pass validation."""
        generator = BaselineStrategyGenerator()
        baselines = generator.get_baseline_strategies()

        # All strategies should be valid after normalization
        for name, strategy in baselines.items():
            assert 2 <= strategy.power <= 5, f"{name} power should be in [2, 5]"
            assert (
                len(strategy.modulus_filters) <= 4
            ), f"{name} should have max 4 filters"
            assert (
                strategy.smoothness_bound in SMALL_PRIMES
            ), f"{name} smoothness_bound should be in SMALL_PRIMES"
            assert (
                1 <= strategy.min_small_prime_hits <= 6
            ), f"{name} min_hits should be in [1, 6]"

    def test_baseline_strategies_are_deterministic(self):
        """Test that baseline strategies are the same each time."""
        generator1 = BaselineStrategyGenerator()
        generator2 = BaselineStrategyGenerator()

        baselines1 = generator1.get_baseline_strategies()
        baselines2 = generator2.get_baseline_strategies()

        # Compare each baseline
        for name in ["conservative", "balanced", "aggressive"]:
            strategy1 = baselines1[name]
            strategy2 = baselines2[name]

            assert strategy1.power == strategy2.power
            assert strategy1.modulus_filters == strategy2.modulus_filters
            assert strategy1.smoothness_bound == strategy2.smoothness_bound
            assert (
                strategy1.min_small_prime_hits == strategy2.min_small_prime_hits
            ), f"{name} should be deterministic"

    def test_conservative_strategy_parameters(self):
        """Test that conservative strategy has strict parameters."""
        generator = BaselineStrategyGenerator()
        conservative = generator.get_baseline_strategies()["conservative"]

        # Conservative should have:
        # - Low power (2-3)
        # - Multiple filters (strict)
        # - High min hits requirement
        assert conservative.power <= 3, "Conservative should use low power"
        assert (
            len(conservative.modulus_filters) >= 2
        ), "Conservative should have multiple filters"
        assert (
            conservative.min_small_prime_hits >= 3
        ), "Conservative should require many small prime hits"

    def test_balanced_strategy_parameters(self):
        """Test that balanced strategy has moderate parameters."""
        generator = BaselineStrategyGenerator()
        balanced = generator.get_baseline_strategies()["balanced"]

        # Balanced should have:
        # - Medium power (3-4)
        # - Moderate filters
        # - Moderate min hits
        assert 3 <= balanced.power <= 4, "Balanced should use medium power"
        assert (
            1 <= len(balanced.modulus_filters) <= 3
        ), "Balanced should have moderate filters"
        assert (
            2 <= balanced.min_small_prime_hits <= 3
        ), "Balanced should have moderate min hits"

    def test_aggressive_strategy_parameters(self):
        """Test that aggressive strategy has permissive parameters."""
        generator = BaselineStrategyGenerator()
        aggressive = generator.get_baseline_strategies()["aggressive"]

        # Aggressive should have:
        # - High power (4-5)
        # - Few filters (permissive)
        # - Low min hits requirement
        assert aggressive.power >= 4, "Aggressive should use high power"
        assert (
            len(aggressive.modulus_filters) <= 2
        ), "Aggressive should have few filters"
        assert (
            aggressive.min_small_prime_hits <= 2
        ), "Aggressive should have low min hits"

    def test_conservative_rejects_more_candidates(self):
        """Test that conservative strategy is more selective than aggressive."""
        generator = BaselineStrategyGenerator()
        baselines = generator.get_baseline_strategies()

        conservative = baselines["conservative"]
        aggressive = baselines["aggressive"]

        # Test on a sample of candidates
        test_n = 961730063
        accepted_conservative = 0
        accepted_aggressive = 0

        for x in range(30000, 31000):  # Sample 1000 candidates
            if conservative(x, test_n):
                accepted_conservative += 1
            if aggressive(x, test_n):
                accepted_aggressive += 1

        # Aggressive should accept more candidates than conservative
        assert (
            accepted_aggressive > accepted_conservative
        ), "Aggressive should accept more candidates"

    def test_baseline_strategies_evaluable(self):
        """Test that baseline strategies can be evaluated in crucible."""
        generator = BaselineStrategyGenerator()
        baselines = generator.get_baseline_strategies()

        crucible = FactorizationCrucible(number_to_factor=961730063)

        # All strategies should be evaluable (not crash)
        for name, strategy in baselines.items():
            metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.05)
            assert (
                metrics.candidate_count >= 0
            ), f"{name} should produce valid candidate count"

    def test_baseline_strategies_produce_nonzero_fitness(self):
        """Test that baseline strategies find at least some candidates."""
        generator = BaselineStrategyGenerator()
        baselines = generator.get_baseline_strategies()

        crucible = FactorizationCrucible(number_to_factor=961730063)

        # At least one baseline should find candidates (sanity check)
        total_candidates = 0
        for name, strategy in baselines.items():
            metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.1)
            total_candidates += metrics.candidate_count

        assert (
            total_candidates > 0
        ), "At least one baseline should find candidates in 0.1s"

    def test_baseline_strategies_different_from_each_other(self):
        """Test that the three baselines are actually different."""
        generator = BaselineStrategyGenerator()
        baselines = generator.get_baseline_strategies()

        conservative = baselines["conservative"]
        balanced = baselines["balanced"]
        aggressive = baselines["aggressive"]

        # They should have different parameters
        assert not (
            conservative.power == balanced.power == aggressive.power
        ), "Powers should differ"

        # At least two should have different min_hits
        hits = [
            conservative.min_small_prime_hits,
            balanced.min_small_prime_hits,
            aggressive.min_small_prime_hits,
        ]
        assert len(set(hits)) >= 2, "Min hits should vary across baselines"
