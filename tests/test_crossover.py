"""Tests for genetic crossover operators."""

import random

from prototype import (
    Strategy,
    blend_modulus_filters,
    crossover_strategies,
)


class TestCrossoverStrategies:
    """Test crossover_strategies function."""

    def test_crossover_creates_valid_strategy(self):
        """Test that crossover produces a valid normalized strategy."""
        parent1 = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        parent2 = Strategy(
            power=4,
            modulus_filters=[(5, [2, 3])],
            smoothness_bound=17,
            min_small_prime_hits=3,
        )

        child = crossover_strategies(parent1, parent2)

        # Should be valid Strategy instance
        assert isinstance(child, Strategy)

        # Should respect constraints (enforced by _normalize)
        assert 2 <= child.power <= 5
        assert len(child.modulus_filters) <= 4
        assert child.smoothness_bound >= 3
        assert 1 <= child.min_small_prime_hits <= 6

    def test_crossover_combines_both_parents(self):
        """Test that child has traits from both parents."""
        parent1 = Strategy(
            power=2,
            modulus_filters=[(3, [0])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        parent2 = Strategy(
            power=5,
            modulus_filters=[(7, [1])],
            smoothness_bound=31,
            min_small_prime_hits=5,
        )

        # Run multiple times to check distribution
        children = [crossover_strategies(parent1, parent2) for _ in range(100)]

        # Child power should be from one of the parents
        powers = {child.power for child in children}
        assert powers.issubset({2, 5}), "Child power should be from parents"

        # Should see both parent values across multiple runs
        assert len(powers) == 2, "Should use both parent power values"

    def test_crossover_with_identical_parents(self):
        """Test crossover handles identical parents gracefully."""
        parent = Strategy(
            power=3,
            modulus_filters=[(7, [0, 2, 4])],
            smoothness_bound=17,
            min_small_prime_hits=3,
        )

        # Crossover with self should return equivalent strategy
        child = crossover_strategies(parent, parent)

        assert child.power == parent.power
        assert child.smoothness_bound == parent.smoothness_bound
        assert child.min_small_prime_hits == parent.min_small_prime_hits
        # Filters might be reordered but should be equivalent
        assert len(child.modulus_filters) == len(parent.modulus_filters)

    def test_crossover_respects_max_filters(self):
        """Test that crossover respects max 4 filters constraint."""
        parent1 = Strategy(
            power=2,
            modulus_filters=[(3, [0]), (5, [1]), (7, [2]), (11, [3])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        parent2 = Strategy(
            power=3,
            modulus_filters=[(13, [4]), (17, [5]), (19, [6]), (23, [7])],
            smoothness_bound=17,
            min_small_prime_hits=3,
        )

        child = crossover_strategies(parent1, parent2)

        # Should never exceed 4 filters
        assert len(child.modulus_filters) <= 4

    def test_crossover_probability_distribution(self):
        """Test that each parent contributes roughly equally."""
        parent1 = Strategy(
            power=2,
            modulus_filters=[(3, [0])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        parent2 = Strategy(
            power=5,
            modulus_filters=[(7, [1])],
            smoothness_bound=31,
            min_small_prime_hits=5,
        )

        # Run many times to check distribution
        power_from_p1 = 0
        bound_from_p1 = 0
        hits_from_p1 = 0
        n_trials = 1000

        for _ in range(n_trials):
            child = crossover_strategies(parent1, parent2)
            if child.power == parent1.power:
                power_from_p1 += 1
            if child.smoothness_bound == parent1.smoothness_bound:
                bound_from_p1 += 1
            if child.min_small_prime_hits == parent1.min_small_prime_hits:
                hits_from_p1 += 1

        # Each parent should contribute roughly 50% (allow 40-60% range)
        assert 400 <= power_from_p1 <= 600, f"Power from p1: {power_from_p1}/1000"
        assert 400 <= bound_from_p1 <= 600, f"Bound from p1: {bound_from_p1}/1000"
        assert 400 <= hits_from_p1 <= 600, f"Hits from p1: {hits_from_p1}/1000"

    def test_crossover_with_different_filter_counts(self):
        """Test crossover when parents have different numbers of filters."""
        parent1 = Strategy(
            power=2,
            modulus_filters=[(3, [0])],  # 1 filter
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        parent2 = Strategy(
            power=3,
            modulus_filters=[(5, [1]), (7, [2]), (11, [3])],  # 3 filters
            smoothness_bound=17,
            min_small_prime_hits=3,
        )

        child = crossover_strategies(parent1, parent2)

        # Should have valid filters
        assert 1 <= len(child.modulus_filters) <= 4
        # All filters should have valid modulus and residues
        for modulus, residues in child.modulus_filters:
            assert modulus >= 2
            assert len(residues) > 0
            assert all(0 <= r < modulus for r in residues)


class TestBlendModulusFilters:
    """Test blend_modulus_filters function."""

    def test_blend_combines_unique_filters(self):
        """Test that blending combines unique filters from both parents."""
        filters1 = [(3, [0, 1])]
        filters2 = [(5, [2, 3])]

        blended = blend_modulus_filters(filters1, filters2, max_filters=4)

        # Should have both unique filters
        assert len(blended) == 2
        moduli = {f[0] for f in blended}
        assert moduli == {3, 5}

    def test_blend_merges_same_modulus(self):
        """Test that filters with same modulus get merged."""
        filters1 = [(7, [0, 1, 2])]
        filters2 = [(7, [3, 4, 5])]

        blended = blend_modulus_filters(filters1, filters2, max_filters=4)

        # Should have single filter with merged residues
        assert len(blended) == 1
        assert blended[0][0] == 7
        # Should contain residues from both parents
        residues = set(blended[0][1])
        assert residues.issuperset({0, 1, 2})
        assert residues.issuperset({3, 4, 5})

    def test_blend_respects_max_filters(self):
        """Test that blending respects maximum filter count."""
        filters1 = [(3, [0]), (5, [1]), (7, [2])]
        filters2 = [(11, [3]), (13, [4]), (17, [5])]

        blended = blend_modulus_filters(filters1, filters2, max_filters=4)

        # Should not exceed max
        assert len(blended) <= 4

    def test_blend_with_empty_filters(self):
        """Test blending when one parent has no filters."""
        filters1 = [(3, [0, 1]), (5, [2, 3])]
        filters2 = []

        blended = blend_modulus_filters(filters1, filters2, max_filters=4)

        # Should use filters from non-empty parent
        assert len(blended) > 0
        assert len(blended) <= 4

    def test_blend_deduplicates_residues(self):
        """Test that blending removes duplicate residues within same modulus."""
        filters1 = [(7, [0, 1, 2])]
        filters2 = [(7, [1, 2, 3])]  # Overlapping residues

        blended = blend_modulus_filters(filters1, filters2, max_filters=4)

        # Should have single filter with deduplicated residues
        assert len(blended) == 1
        residues = blended[0][1]
        # Should be sorted and unique
        assert residues == sorted(set(residues))
        # Should contain union of both sets
        assert set(residues) == {0, 1, 2, 3}

    def test_blend_prioritizes_smaller_moduli(self):
        """Test that blending prioritizes smaller moduli when limiting."""
        # Create many filters to force limiting
        filters1 = [(3, [0]), (7, [1]), (13, [2])]
        filters2 = [(5, [3]), (11, [4]), (17, [5])]

        blended = blend_modulus_filters(filters1, filters2, max_filters=3)

        # Should keep smaller moduli (better for filtering)
        moduli = {f[0] for f in blended}
        # Should prefer smaller primes (3, 5, 7) over larger (11, 13, 17)
        assert 3 in moduli or 5 in moduli or 7 in moduli


class TestCrossoverIntegration:
    """Integration tests for crossover in evolutionary context."""

    def test_crossover_maintains_fitness_capability(self):
        """Test that crossed strategies can still evaluate fitness."""
        from prototype import FactorizationCrucible

        crucible = FactorizationCrucible(961730063)

        parent1 = Strategy(
            power=2,
            modulus_filters=[(2, [0]), (3, [0])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        parent2 = Strategy(
            power=3,
            modulus_filters=[(5, [0]), (7, [0])],
            smoothness_bound=17,
            min_small_prime_hits=3,
        )

        child = crossover_strategies(parent1, parent2)

        # Child should be evaluable (not crash)
        fitness = crucible.evaluate_strategy(child, duration_seconds=0.05)
        assert isinstance(fitness, int)
        assert fitness >= 0

    def test_multiple_crossovers_create_diversity(self):
        """Test that repeated crossovers create diverse offspring."""
        parent1 = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        parent2 = Strategy(
            power=4,
            modulus_filters=[(7, [2, 3])],
            smoothness_bound=23,
            min_small_prime_hits=4,
        )

        # Create multiple offspring
        offspring = [crossover_strategies(parent1, parent2) for _ in range(20)]

        # Should see variety in powers
        powers = {child.power for child in offspring}
        assert len(powers) > 1, "Should create diverse power values"

        # Should see variety in other parameters
        bounds = {child.smoothness_bound for child in offspring}
        assert len(bounds) > 1, "Should create diverse smoothness bounds"
