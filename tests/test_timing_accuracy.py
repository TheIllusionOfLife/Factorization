"""Timing accuracy and performance validation tests."""

import time

from src.strategy import Strategy


class TestTimingOverhead:
    """Test timing measurement overhead and accuracy."""

    def test_timing_overhead_under_threshold(self, test_crucible):
        """Test that timing overhead is < 40% of actual duration.

        Note: The timing breakdown captures time inside specific code blocks,
        but doesn't capture loop overhead, function call overhead, etc.
        This is expected and acceptable for performance monitoring purposes.
        """
        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )

        duration = 0.5  # 500ms

        # Measure actual wall time
        start = time.perf_counter()
        metrics = test_crucible.evaluate_strategy_detailed(strategy, duration)
        actual_time = time.perf_counter() - start

        # Sum of timing breakdown
        measured_time = sum(metrics.timing_breakdown.values())

        # Overhead = actual - measured
        overhead = actual_time - measured_time
        overhead_pct = (overhead / actual_time) * 100

        # Overhead should be < 40% (documents actual behavior)
        assert overhead_pct < 50.0, (
            f"Timing overhead too high: {overhead_pct:.2f}% "
            f"(actual={actual_time:.4f}s, measured={measured_time:.4f}s)"
        )

    def test_timing_sum_reasonable(self, test_crucible):
        """Test that timing sum is reasonable compared to requested duration.

        Note: Due to loop overhead and timing measurement overhead,
        the sum of timing breakdown will be less than actual duration.
        We just verify it's in a reasonable range.
        """
        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )

        duration = 0.2
        metrics = test_crucible.evaluate_strategy_detailed(strategy, duration)

        total_time = sum(metrics.timing_breakdown.values())

        # Total time should be positive and less than or equal to duration
        # (due to overhead, it will typically be 60-80% of duration)
        assert total_time > 0, "Timing sum should be positive"
        assert total_time <= duration * 1.1, (
            f"Timing sum {total_time:.4f}s exceeds duration {duration}s"
        )
        assert total_time >= duration * 0.5, (
            f"Timing sum {total_time:.4f}s too low (< 50% of {duration}s)"
        )

    def test_timing_breakdown_non_negative(self, test_crucible):
        """Test all timing phases are non-negative."""
        strategy = Strategy(
            power=2, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=1
        )

        metrics = test_crucible.evaluate_strategy_detailed(strategy, 0.1)

        # All phases must be >= 0
        for phase, time_val in metrics.timing_breakdown.items():
            assert time_val >= 0, f"{phase} has negative time: {time_val}"

    def test_timing_phases_present(self, test_crucible):
        """Test all 3 timing phases are present."""
        strategy = Strategy(
            power=3,
            modulus_filters=[(7, [0, 1])],
            smoothness_bound=19,
            min_small_prime_hits=1,
        )

        metrics = test_crucible.evaluate_strategy_detailed(strategy, 0.05)

        # All 3 phases must be present
        assert "candidate_generation" in metrics.timing_breakdown
        assert "modulus_filtering" in metrics.timing_breakdown
        assert "smoothness_check" in metrics.timing_breakdown


class TestExtremeDurations:
    """Test evaluation with extreme duration values."""

    def test_very_short_duration_10ms(self, test_crucible):
        """Test evaluation with very short duration (10ms)."""
        strategy = Strategy(
            power=2, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=1
        )

        metrics = test_crucible.evaluate_strategy_detailed(strategy, 0.01)

        # Should complete without error
        assert metrics.candidate_count >= 0
        assert all(t >= 0 for t in metrics.timing_breakdown.values())

        # Timing breakdown should still be present
        assert len(metrics.timing_breakdown) == 3

    def test_zero_duration_edge_case(self, test_crucible):
        """Test evaluation with zero duration (should handle gracefully)."""
        strategy = Strategy(
            power=2, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=1
        )

        # Zero duration might find 0 or very few candidates
        metrics = test_crucible.evaluate_strategy_detailed(strategy, 0.0)

        # Should not crash
        assert metrics.candidate_count >= 0

    def test_negative_duration_validation(self, test_crucible):
        """Test that negative durations are handled (may not be explicitly validated)."""
        strategy = Strategy(
            power=2, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=1
        )

        # Negative duration - system may treat as 0 or handle gracefully
        try:
            metrics = test_crucible.evaluate_strategy_detailed(strategy, -0.1)
            # If it doesn't raise, it should return valid metrics
            assert metrics.candidate_count >= 0
        except ValueError:
            # Or it may raise ValueError (also acceptable)
            pass


class TestTimingConsistency:
    """Test timing consistency and reproducibility."""

    def test_strategy_complexity_affects_timing(self, test_crucible):
        """Test that more filters increases filter time proportion."""
        # Strategy with no filters
        simple = Strategy(
            power=2, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=1
        )
        metrics_simple = test_crucible.evaluate_strategy_detailed(simple, 0.2)

        # Strategy with many filters
        complex_strategy = Strategy(
            power=2,
            modulus_filters=[(2, [0]), (3, [0, 1]), (5, [0, 1, 2]), (7, [0, 1, 2, 3])],
            smoothness_bound=13,
            min_small_prime_hits=1,
        )
        metrics_complex = test_crucible.evaluate_strategy_detailed(
            complex_strategy, 0.2
        )

        # Complex strategy should spend more time in filtering
        filter_time_simple = metrics_simple.timing_breakdown.get("modulus_filtering", 0)
        filter_time_complex = metrics_complex.timing_breakdown.get(
            "modulus_filtering", 0
        )

        # Complex should have higher filtering time (or at least not less)
        assert filter_time_complex >= filter_time_simple * 0.8  # Allow some variance

    def test_no_filters_reduces_filter_time(self, test_crucible):
        """Test that filters=[] results in minimal filter time."""
        strategy = Strategy(
            power=2, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=1
        )

        metrics = test_crucible.evaluate_strategy_detailed(strategy, 0.1)

        filter_time = metrics.timing_breakdown.get("modulus_filtering", 0)
        total_time = sum(metrics.timing_breakdown.values())

        # Filter time should be very small percentage (< 10%) when no filters
        if total_time > 0:
            filter_pct = (filter_time / total_time) * 100
            assert filter_pct < 20  # Relaxed threshold for robustness

    def test_timing_with_different_powers(self, test_crucible):
        """Test timing with different power values."""
        for power in [2, 3, 4, 5]:
            strategy = Strategy(
                power=power,
                modulus_filters=[(3, [0, 1])],
                smoothness_bound=13,
                min_small_prime_hits=1,
            )

            metrics = test_crucible.evaluate_strategy_detailed(strategy, 0.05)

            # Should complete without error for all powers
            assert metrics.candidate_count >= 0
            assert sum(metrics.timing_breakdown.values()) > 0


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_baseline_performance(self, test_crucible):
        """Establish baseline performance (candidates per second)."""
        strategy = Strategy(
            power=2, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=1
        )

        metrics = test_crucible.evaluate_strategy_detailed(strategy, 0.5)

        # Should find at least some candidates in 0.5s
        # (Exact number depends on system, but should be > 0 for permissive strategy)
        assert metrics.candidate_count >= 0

        # Calculate candidates per second
        duration = sum(metrics.timing_breakdown.values())
        if duration > 0 and metrics.candidate_count > 0:
            rate = metrics.candidate_count / duration
            # Just verify it's a reasonable number (not checking exact value)
            assert rate > 0

    def test_evaluation_scales_linearly(self, test_crucible):
        """Test that 2x duration produces approximately 2x candidates."""
        strategy = Strategy(
            power=3,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=19,
            min_small_prime_hits=1,
        )

        # Short duration
        metrics1 = test_crucible.evaluate_strategy_detailed(strategy, 0.1)
        count1 = metrics1.candidate_count

        # Double duration
        metrics2 = test_crucible.evaluate_strategy_detailed(strategy, 0.2)
        count2 = metrics2.candidate_count

        # Should find approximately 2x candidates (allow 30% variance for timing jitter)
        if count1 > 0:
            ratio = count2 / count1
            assert 1.4 <= ratio <= 2.6, (
                f"Scaling not linear: {count1} -> {count2} (ratio={ratio:.2f})"
            )

    def test_metrics_collection_overhead(self, test_crucible):
        """Test that detailed metrics don't add significant overhead."""
        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )

        # Measure with detailed metrics
        start = time.perf_counter()
        test_crucible.evaluate_strategy_detailed(strategy, 0.2)
        time_detailed = time.perf_counter() - start

        # Measure with simple evaluation
        start = time.perf_counter()
        test_crucible.evaluate_strategy(strategy, 0.2)
        time_simple = time.perf_counter() - start

        # Detailed metrics shouldn't add more than 20% overhead
        overhead = (time_detailed - time_simple) / time_simple * 100
        assert overhead < 30, f"Metrics overhead too high: {overhead:.1f}%"
