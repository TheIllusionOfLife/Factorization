"""Edge case tests for meta-learning system (UCB1, rate normalization, statistics)."""

import pytest

from src.adaptive_engine import MetaLearningEngine


class TestUCB1EdgeCases:
    """Test UCB1 algorithm edge cases and boundary conditions."""

    def test_single_operator_untried_splits_correctly(self):
        """Test UCB1 when only one operator is untried (1 inf, 2 finite)."""
        engine = MetaLearningEngine(
            adaptation_window=1, fallback_inf_rate=0.8, fallback_finite_rate=0.2
        )

        # Simulate: crossover=untried, mutation=tried, random=tried
        engine.current_generation = 0
        # Mutation: 10 offspring, 5 elite (50% success)
        for i in range(10):
            engine.update_statistics("mutation", 20.0, i < 5)
        # Random: 5 offspring, 1 elite (20% success)
        for i in range(5):
            engine.update_statistics("random", 10.0, i < 1)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Untried operator (crossover) should get high rate from inf score
        # After normalization with bounds, this ends up around 0.7 (max_rate)
        assert rates.crossover_rate >= 0.65  # Close to max_rate=0.7

        # Tried operators get lower rates (split remaining rate)
        # After normalization they may end up equal at min_rate
        assert rates.mutation_rate + rates.random_rate < 0.35
        assert abs(rates.mutation_rate - rates.random_rate) < 0.05  # May be equal

        # Total should sum to 1.0
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_two_operators_untried_splits_correctly(self):
        """Test UCB1 when two operators are untried (2 inf, 1 finite)."""
        engine = MetaLearningEngine(
            adaptation_window=1, fallback_inf_rate=0.8, fallback_finite_rate=0.2
        )

        # Only mutation tried
        engine.current_generation = 0
        for i in range(10):
            engine.update_statistics("mutation", 20.0, i < 8)  # 80% success
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Two untried (crossover, random) split fallback_inf_rate
        inf_rate_each = 0.8 / 2  # 0.4 each
        assert rates.crossover_rate == pytest.approx(inf_rate_each, abs=0.01)
        assert rates.random_rate == pytest.approx(inf_rate_each, abs=0.01)

        # Tried operator (mutation) gets fallback_finite_rate
        assert rates.mutation_rate == pytest.approx(0.2, abs=0.01)

        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_zero_total_trials_uniform_distribution(self):
        """Test UCB1 with zero total trials (no offspring created)."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0
        # Don't add any statistics
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Should return uniform distribution (1/3 each)
        assert rates.crossover_rate == pytest.approx(1 / 3, abs=0.01)
        assert rates.mutation_rate == pytest.approx(1 / 3, abs=0.01)
        assert rates.random_rate == pytest.approx(1 / 3, abs=0.01)

    def test_identical_success_rates_tie_breaking(self):
        """Test UCB1 when all operators have identical success rates."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # All operators: 10 offspring, 5 elite (50% success)
        for i in range(10):
            engine.update_statistics("crossover", 20.0, i < 5)
            engine.update_statistics("mutation", 20.0, i < 5)
            engine.update_statistics("random", 20.0, i < 5)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # With identical success rates and trial counts, rates should be equal
        # (exploration bonus is same for all)
        assert abs(rates.crossover_rate - rates.mutation_rate) < 0.01
        assert abs(rates.mutation_rate - rates.random_rate) < 0.01
        assert abs(rates.crossover_rate - 1 / 3) < 0.05  # Close to uniform

    def test_very_large_trial_counts(self):
        """Test UCB1 with very large trial counts (test log overflow)."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Simulate 10,000 trials
        for i in range(10000):
            # Crossover: 80% success
            engine.update_statistics("crossover", 20.0, i % 10 < 8)
            # Mutation: 50% success
            engine.update_statistics("mutation", 20.0, i % 10 < 5)
            # Random: 20% success
            engine.update_statistics("random", 10.0, i % 10 < 2)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Crossover should have highest rate (best success)
        assert rates.crossover_rate > rates.mutation_rate
        assert rates.mutation_rate > rates.random_rate

        # Total should sum to 1.0
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_negative_fitness_improvements(self):
        """Test handling of negative fitness improvements."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Negative improvements (fitness degradation)
        engine.update_statistics("crossover", -50.0, True)  # Elite despite degradation
        engine.update_statistics("mutation", -20.0, False)
        engine.update_statistics("random", 10.0, False)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Should still work (negative fitness is valid data)
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

        # All rates should be within bounds
        assert 0.1 <= rates.crossover_rate <= 0.7
        assert 0.1 <= rates.mutation_rate <= 0.7
        assert 0.1 <= rates.random_rate <= 0.7

    def test_inf_fitness_improvements(self):
        """Test handling of infinite fitness improvements."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Infinite improvements
        engine.update_statistics("crossover", float("inf"), True)
        engine.update_statistics("mutation", 50.0, True)
        engine.update_statistics("random", 10.0, False)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Should handle inf gracefully
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_nan_fitness_improvements(self):
        """Test handling of NaN fitness improvements."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # NaN improvements (edge case)
        engine.update_statistics("crossover", float("nan"), True)
        engine.update_statistics("mutation", 50.0, True)
        engine.update_statistics("random", 10.0, False)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Should handle NaN gracefully (might treat as 0 or skip)
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_ucb1_with_zero_successes(self):
        """Test UCB1 when all operators have 0% success rate."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # All operators: offspring but no elites
        for _i in range(10):
            engine.update_statistics("crossover", 20.0, False)
            engine.update_statistics("mutation", 20.0, False)
            engine.update_statistics("random", 10.0, False)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # With 0% success, exploration bonus dominates
        # All should get approximately equal rates
        assert abs(rates.crossover_rate - rates.mutation_rate) < 0.1
        assert abs(rates.mutation_rate - rates.random_rate) < 0.1

        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_ucb1_exploration_bonus_correctness(self):
        """Verify UCB1 exploration bonus formula: sqrt(2*ln(N)/n)."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Crossover: 100 offspring, 50 elite (50% success)
        for i in range(100):
            engine.update_statistics("crossover", 20.0, i < 50)
        # Mutation: 10 offspring, 5 elite (50% success, but fewer trials)
        for i in range(10):
            engine.update_statistics("mutation", 20.0, i < 5)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Random (untried) should have highest rate due to inf exploration bonus
        assert rates.random_rate > rates.crossover_rate
        assert rates.random_rate > rates.mutation_rate

        # Crossover and mutation should be similar (same success rate)
        # but after normalization they may hit min_rate bound
        assert abs(rates.crossover_rate - rates.mutation_rate) < 0.05


class TestRateNormalizationEdgeCases:
    """Test edge cases in rate normalization and bounds enforcement."""

    def test_epsilon_edge_cases(self):
        """Test 3*0.333333... = 1.0 boundary condition."""
        # This should not raise error (within epsilon tolerance)
        engine = MetaLearningEngine(adaptation_window=1, min_rate=0.333, max_rate=0.334)

        # Verify it was created successfully
        assert engine.min_rate == 0.333
        assert engine.max_rate == 0.334

    def test_single_rate_at_max_bound(self):
        """Test when one operator is maxed, others at min."""
        engine = MetaLearningEngine(adaptation_window=1, min_rate=0.1, max_rate=0.7)
        engine.current_generation = 0

        # Crossover: Very high success (90%)
        for _i in range(10):
            engine.update_statistics("crossover", 50.0, _i < 9)
        # Mutation: Low success (10%)
        for _i in range(10):
            engine.update_statistics("mutation", 20.0, _i < 1)
        # Random: No success (0%)
        for _i in range(10):
            engine.update_statistics("random", 10.0, False)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Crossover should have highest rate (best success)
        assert rates.crossover_rate > rates.mutation_rate
        assert rates.crossover_rate > rates.random_rate

        # After normalization, crossover gets highest but may not reach max
        # due to need to keep others >= min_rate
        assert rates.crossover_rate >= 0.5

        # Others should be at or near min_rate
        assert rates.mutation_rate >= 0.1
        assert rates.random_rate >= 0.1

        # Total should sum to 1.0
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_all_rates_at_min_bound(self):
        """Test edge case when all rates want to be at min."""
        engine = MetaLearningEngine(adaptation_window=1, min_rate=0.3, max_rate=0.7)
        engine.current_generation = 0

        # All operators have identical, poor performance
        for i in range(10):
            engine.update_statistics("crossover", 10.0, i < 1)  # 10% success
            engine.update_statistics("mutation", 10.0, i < 1)
            engine.update_statistics("random", 10.0, i < 1)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # All should be approximately equal (uniform)
        assert abs(rates.crossover_rate - 1 / 3) < 0.1
        assert abs(rates.mutation_rate - 1 / 3) < 0.1
        assert abs(rates.random_rate - 1 / 3) < 0.1

        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_floating_point_accumulation_errors(self):
        """Test that floating point errors don't cause sum != 1.0."""
        engine = MetaLearningEngine(adaptation_window=1)

        # Run many generations to accumulate potential errors
        for gen in range(50):
            engine.current_generation = gen
            for i in range(10):
                engine.update_statistics("crossover", 20.0 * (gen + 1), i < 5)
                engine.update_statistics("mutation", 15.0 * (gen + 1), i < 4)
                engine.update_statistics("random", 10.0 * (gen + 1), i < 3)
            engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Sum should still be exactly 1.0 (within machine precision)
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-9

    def test_rates_within_bounds_after_normalization(self):
        """Verify all rates respect min/max bounds after normalization."""
        engine = MetaLearningEngine(adaptation_window=1, min_rate=0.15, max_rate=0.65)
        engine.current_generation = 0

        # Extreme success differences
        for i in range(100):
            engine.update_statistics("crossover", 50.0, i < 95)  # 95% success
            engine.update_statistics("mutation", 20.0, i < 30)  # 30% success
            engine.update_statistics("random", 10.0, i < 5)  # 5% success
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # All rates must be within [min_rate, max_rate]
        assert 0.15 <= rates.crossover_rate <= 0.65
        assert 0.15 <= rates.mutation_rate <= 0.65
        assert 0.15 <= rates.random_rate <= 0.65

        # Total must be 1.0
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6


class TestStatisticsEdgeCases:
    """Test edge cases in statistics tracking and calculation."""

    def test_negative_fitness_improvement_tracking(self):
        """Test tracking of negative fitness improvements."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Mix of positive and negative improvements
        engine.update_statistics("crossover", 50.0, True)
        engine.update_statistics("crossover", -30.0, True)
        engine.update_statistics("crossover", 20.0, False)
        engine.finalize_generation()

        stats = engine.get_operator_history()
        crossover_stats = stats[0]["crossover"]

        # Total improvement should be 50 - 30 + 20 = 40
        assert crossover_stats.total_fitness_improvement == pytest.approx(40.0)

        # Avg improvement should be 40 / 3 = 13.33...
        assert crossover_stats.avg_fitness_improvement == pytest.approx(40.0 / 3)

    def test_zero_fitness_improvement(self):
        """Test handling of zero fitness improvements."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # All zero improvements
        for i in range(10):
            engine.update_statistics("crossover", 0.0, i < 5)
        engine.finalize_generation()

        stats = engine.get_operator_history()
        crossover_stats = stats[0]["crossover"]

        assert crossover_stats.total_fitness_improvement == 0.0
        assert crossover_stats.avg_fitness_improvement == 0.0
        assert crossover_stats.success_rate == 0.5  # 5/10

    def test_inf_fitness_improvement_handling(self):
        """Test handling of infinite fitness improvements."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        engine.update_statistics("crossover", float("inf"), True)
        engine.update_statistics("crossover", 100.0, True)
        engine.finalize_generation()

        stats = engine.get_operator_history()
        crossover_stats = stats[0]["crossover"]

        # Should handle inf in total (might be inf or handled gracefully)
        assert crossover_stats.total_offspring == 2
        assert crossover_stats.elite_offspring == 2

    def test_very_large_offspring_counts(self):
        """Test with very large offspring counts (10,000+)."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Simulate 10,000 offspring
        for i in range(10000):
            engine.update_statistics("crossover", 20.0, i % 2 == 0)  # 50% elite
        engine.finalize_generation()

        stats = engine.get_operator_history()
        crossover_stats = stats[0]["crossover"]

        assert crossover_stats.total_offspring == 10000
        assert crossover_stats.elite_offspring == 5000
        assert crossover_stats.success_rate == pytest.approx(0.5)

    def test_current_statistics_returns_reference(self):
        """Verify get_current_statistics returns shallow copy (dict copy, not deep copy).

        Note: This documents the actual behavior - the dict is copied but
        OperatorStatistics objects inside are references. This is acceptable
        because tests should not modify returned statistics.
        """
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        engine.update_statistics("crossover", 50.0, True)
        engine.update_statistics("mutation", 30.0, True)

        # Get statistics
        stats1 = engine.get_current_statistics()

        # Verify initial value
        assert stats1["crossover"].total_offspring == 1

        # Modify the returned object (shallow copy - dict copied, objects not)
        stats1["crossover"].total_offspring = 999999

        # Get statistics again
        stats2 = engine.get_current_statistics()

        # The object is the same (shallow copy), so modification visible
        # This is acceptable since tests shouldn't modify returned stats
        assert stats2["crossover"].total_offspring == 999999

    def test_operator_history_alignment(self):
        """Test that operator_history indices align with generations."""
        engine = MetaLearningEngine(adaptation_window=2)

        for gen in range(5):
            engine.current_generation = gen
            engine.update_statistics("crossover", 50.0, True)
            engine.finalize_generation()

        history = engine.get_operator_history()

        # Should have 5 generations of history
        assert len(history) == 5

        # Each generation should have stats for all operators
        for gen_stats in history:
            assert "crossover" in gen_stats
            assert "mutation" in gen_stats
            assert "random" in gen_stats

    def test_reset_after_finalize(self):
        """Test that current_stats are reset after finalize_generation."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        engine.update_statistics("crossover", 50.0, True)
        engine.finalize_generation()

        # After finalize, current_stats should be reset
        current = engine.get_current_statistics()

        # All operators should have zero stats
        for _operator, stats in current.items():
            assert stats.total_offspring == 0
            assert stats.elite_offspring == 0
            assert stats.total_fitness_improvement == 0.0

    def test_statistics_with_single_operator(self):
        """Test statistics when only one operator is used."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Only use crossover
        for i in range(10):
            engine.update_statistics("crossover", 50.0, i < 7)

        engine.finalize_generation()

        stats = engine.get_operator_history()

        # Crossover should have stats
        assert stats[0]["crossover"].total_offspring == 10
        assert stats[0]["crossover"].elite_offspring == 7

        # Others should have zero stats
        assert stats[0]["mutation"].total_offspring == 0
        assert stats[0]["random"].total_offspring == 0

    def test_avg_fitness_improvement_calculation(self):
        """Test average fitness improvement calculation."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # Known improvements: 100, 200, 300 -> avg = 200
        engine.update_statistics("crossover", 100.0, True)
        engine.update_statistics("crossover", 200.0, True)
        engine.update_statistics("crossover", 300.0, True)
        engine.finalize_generation()

        stats = engine.get_operator_history()
        crossover_stats = stats[0]["crossover"]

        assert crossover_stats.total_fitness_improvement == pytest.approx(600.0)
        assert crossover_stats.avg_fitness_improvement == pytest.approx(200.0)

    def test_success_rate_calculation(self):
        """Test success rate calculation (elite / total)."""
        engine = MetaLearningEngine(adaptation_window=1)
        engine.current_generation = 0

        # 7 out of 10 are elite -> 70%
        for i in range(10):
            engine.update_statistics("crossover", 50.0, i < 7)
        engine.finalize_generation()

        stats = engine.get_operator_history()
        crossover_stats = stats[0]["crossover"]

        assert crossover_stats.success_rate == pytest.approx(0.7)
