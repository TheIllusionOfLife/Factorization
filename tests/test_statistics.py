"""
Tests for statistical analysis utilities.

Following TDD: These tests are written BEFORE implementation.
All tests should FAIL initially (RED phase).
"""

import pytest

from src.statistics import (
    ComparisonResult,
    ConvergenceDetector,
    StatisticalAnalyzer,
)


class TestStatisticalAnalyzer:
    """Test statistical comparison functionality."""

    def test_analyzer_initialization(self):
        """Test that analyzer can be created with default alpha."""
        analyzer = StatisticalAnalyzer()
        assert analyzer.alpha == 0.05

    def test_analyzer_custom_alpha(self):
        """Test analyzer with custom significance level."""
        analyzer = StatisticalAnalyzer(alpha=0.01)
        assert analyzer.alpha == 0.01

    def test_compare_identical_distributions(self):
        """Test that identical distributions have high p-value (not significant)."""
        analyzer = StatisticalAnalyzer()
        scores1 = [100.0, 105.0, 98.0, 102.0, 101.0]
        scores2 = [100.0, 105.0, 98.0, 102.0, 101.0]

        result = analyzer.compare_fitness_distributions(scores1, scores2)

        assert result.p_value > 0.05, (
            "Identical distributions should not be significant"
        )
        assert not result.is_significant
        assert abs(result.evolved_mean - result.baseline_mean) < 0.1
        assert abs(result.effect_size) < 0.2  # Very small effect

    def test_compare_significantly_different_distributions(self):
        """Test that clearly different distributions show significance."""
        analyzer = StatisticalAnalyzer()
        # Evolved scores much higher than baseline
        evolved = [150.0, 155.0, 148.0, 152.0, 151.0]
        baseline = [100.0, 105.0, 98.0, 102.0, 101.0]

        result = analyzer.compare_fitness_distributions(evolved, baseline)

        assert result.evolved_mean > result.baseline_mean
        assert result.p_value < 0.05, "Clear difference should be significant"
        assert result.is_significant
        assert result.effect_size > 1.0  # Large effect size

    def test_compare_slight_difference(self):
        """Test distributions with small difference (may not be significant)."""
        analyzer = StatisticalAnalyzer()
        # Small difference with overlap
        evolved = [105.0, 108.0, 102.0, 107.0, 106.0]
        baseline = [100.0, 105.0, 98.0, 102.0, 101.0]

        result = analyzer.compare_fitness_distributions(evolved, baseline)

        assert result.evolved_mean > result.baseline_mean
        # Effect size should be positive (evolved better)
        assert result.effect_size > 0

    def test_cohen_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        analyzer = StatisticalAnalyzer()
        # Known case: one standard deviation apart
        group1 = [10.0, 12.0, 11.0, 10.5, 11.5]  # mean ~11
        group2 = [5.0, 7.0, 6.0, 5.5, 6.5]  # mean ~6

        result = analyzer.compare_fitness_distributions(group1, group2)

        # Should have large effect size (>0.8)
        assert result.effect_size > 0.8

    def test_confidence_interval_contains_difference(self):
        """Test that confidence interval contains the true difference."""
        analyzer = StatisticalAnalyzer()
        evolved = [150.0, 155.0, 148.0, 152.0, 151.0]  # mean ~151
        baseline = [100.0, 105.0, 98.0, 102.0, 101.0]  # mean ~101

        result = analyzer.compare_fitness_distributions(evolved, baseline)

        true_diff = result.evolved_mean - result.baseline_mean
        ci_lower, ci_upper = result.confidence_interval

        # CI should contain the difference
        assert ci_lower <= true_diff <= ci_upper
        # CI should be positive (evolved better than baseline)
        assert ci_lower > 0

    def test_comparison_result_interpretation(self):
        """Test human-readable interpretation of results."""
        analyzer = StatisticalAnalyzer()
        evolved = [150.0, 155.0, 148.0]
        baseline = [100.0, 105.0, 98.0]

        result = analyzer.compare_fitness_distributions(evolved, baseline)
        interpretation = result.interpret()

        # Should return a string
        assert isinstance(interpretation, str)
        assert len(interpretation) > 20  # Meaningful description
        # Should mention significance if p < 0.05
        if result.is_significant:
            assert "significant" in interpretation.lower()

    def test_empty_lists_raise_error(self):
        """Test that empty fitness lists raise appropriate error."""
        analyzer = StatisticalAnalyzer()

        with pytest.raises((ValueError, IndexError)):
            analyzer.compare_fitness_distributions([], [100.0])

        with pytest.raises((ValueError, IndexError)):
            analyzer.compare_fitness_distributions([100.0], [])

    def test_single_value_lists(self):
        """Test handling of lists with single values."""
        analyzer = StatisticalAnalyzer()
        # Single values should still compute (though not very meaningful)
        result = analyzer.compare_fitness_distributions([150.0], [100.0])

        assert result.evolved_mean == 150.0
        assert result.baseline_mean == 100.0

    def test_unequal_length_lists(self):
        """Test that lists of different lengths are handled correctly."""
        analyzer = StatisticalAnalyzer()
        evolved = [150.0, 155.0, 148.0, 152.0, 151.0]
        baseline = [100.0, 105.0, 98.0]  # Shorter list

        # Should still work (Welch's t-test handles unequal sizes)
        result = analyzer.compare_fitness_distributions(evolved, baseline)

        assert result.evolved_mean > result.baseline_mean
        assert result.p_value >= 0  # Valid p-value

    def test_negative_fitness_values(self):
        """Test handling of negative fitness values (edge case)."""
        analyzer = StatisticalAnalyzer()
        evolved = [-10.0, -8.0, -12.0]
        baseline = [-50.0, -52.0, -48.0]

        result = analyzer.compare_fitness_distributions(evolved, baseline)

        # Less negative is better
        assert result.evolved_mean > result.baseline_mean


class TestConvergenceDetector:
    """Test convergence detection functionality."""

    def test_detector_initialization(self):
        """Test convergence detector initialization."""
        detector = ConvergenceDetector(window_size=5, threshold=0.05)
        assert detector.window_size == 5
        assert detector.threshold == 0.05

    def test_converged_flat_fitness(self):
        """Test detection of converged (flat) fitness history."""
        detector = ConvergenceDetector(window_size=3, threshold=0.01)
        # Flat fitness for last 3 generations
        fitness_history = [100, 200, 300, 500, 505, 502, 503]

        assert detector.has_converged(fitness_history)

    def test_not_converged_still_improving(self):
        """Test that improving fitness is not considered converged."""
        detector = ConvergenceDetector(window_size=3, threshold=0.01)
        # Still improving significantly
        fitness_history = [100, 200, 300, 400, 500, 600, 700]

        assert not detector.has_converged(fitness_history)

    def test_not_converged_insufficient_data(self):
        """Test that insufficient history returns not converged."""
        detector = ConvergenceDetector(window_size=5, threshold=0.05)
        # Only 3 generations, need 5
        fitness_history = [100, 200, 300]

        assert not detector.has_converged(fitness_history)

    def test_converged_with_small_noise(self):
        """Test that small variations don't prevent convergence."""
        detector = ConvergenceDetector(window_size=4, threshold=0.01)
        # Mostly flat with tiny noise
        fitness_history = [100, 200, 500, 502, 501, 503, 502.5]

        assert detector.has_converged(fitness_history)

    def test_not_converged_with_large_variance(self):
        """Test that large variance prevents convergence."""
        detector = ConvergenceDetector(
            window_size=3, threshold=0.001
        )  # Very strict threshold
        # High variance in recent generations
        fitness_history = [100, 200, 500, 450, 550, 480, 520]

        # With strict threshold, this should not converge due to variance
        assert not detector.has_converged(fitness_history)

    def test_generations_to_convergence(self):
        """Test finding the generation where convergence first occurred."""
        detector = ConvergenceDetector(window_size=3, threshold=0.01)
        # Converges at generation 4 (index 4, window [2,3,4])
        fitness_history = [100, 200, 500, 502, 501, 503]  # Flat from index 2

        gen = detector.generations_to_convergence(fitness_history)

        # Should return generation index where convergence started
        assert gen is not None
        assert gen >= detector.window_size - 1

    def test_generations_to_convergence_never(self):
        """Test when fitness never converges."""
        detector = ConvergenceDetector(window_size=3, threshold=0.01)
        # Always improving, never flat
        fitness_history = [100, 200, 300, 400, 500, 600]

        gen = detector.generations_to_convergence(fitness_history)

        assert gen is None

    def test_convergence_with_zero_mean(self):
        """Test edge case where fitness mean is near zero."""
        detector = ConvergenceDetector(window_size=3, threshold=0.1)
        # Near-zero fitness (edge case for relative variance)
        fitness_history = [1.0, 0.5, 0.1, 0.05, 0.05, 0.05]

        # Should handle gracefully (not crash)
        result = detector.has_converged(fitness_history)
        # Python bool, not numpy bool
        assert isinstance(result, bool)

    def test_convergence_custom_threshold(self):
        """Test that threshold affects convergence detection."""
        # Very strict threshold
        strict_detector = ConvergenceDetector(window_size=3, threshold=0.000001)
        # Very loose threshold
        loose_detector = ConvergenceDetector(window_size=3, threshold=0.5)

        fitness_history_variable = [100, 200, 500, 510, 495, 505]  # More variation
        fitness_history_flat = [100, 200, 500, 500.1, 500.05, 500.02]  # Very flat

        # Strict detector should not converge on variable data
        assert not strict_detector.has_converged(fitness_history_variable)
        # Loose detector should converge on both
        assert loose_detector.has_converged(fitness_history_variable)
        assert loose_detector.has_converged(fitness_history_flat)


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult."""
        result = ComparisonResult(
            evolved_mean=150.0,
            baseline_mean=100.0,
            t_statistic=3.5,
            p_value=0.003,
            effect_size=1.2,
            is_significant=True,
            confidence_interval=(30.0, 70.0),
        )

        assert result.evolved_mean == 150.0
        assert result.baseline_mean == 100.0
        assert result.is_significant

    def test_comparison_result_interpret_significant(self):
        """Test interpretation of significant result."""
        result = ComparisonResult(
            evolved_mean=150.0,
            baseline_mean=100.0,
            t_statistic=3.5,
            p_value=0.003,
            effect_size=1.2,
            is_significant=True,
            confidence_interval=(30.0, 70.0),
        )

        interpretation = result.interpret()

        assert "significant" in interpretation.lower()
        assert "+50.0%" in interpretation  # Shows improvement percentage
        assert "large" in interpretation.lower()  # Effect size description

    def test_comparison_result_interpret_not_significant(self):
        """Test interpretation of non-significant result."""
        result = ComparisonResult(
            evolved_mean=105.0,
            baseline_mean=100.0,
            t_statistic=0.8,
            p_value=0.4,
            effect_size=0.2,
            is_significant=False,
            confidence_interval=(-10.0, 20.0),
        )

        interpretation = result.interpret()

        assert "no statistically significant" in interpretation.lower()

    def test_comparison_result_improvement_percentage(self):
        """Test that result can show improvement percentage."""
        result = ComparisonResult(
            evolved_mean=150.0,
            baseline_mean=100.0,
            t_statistic=3.5,
            p_value=0.003,
            effect_size=1.2,
            is_significant=True,
            confidence_interval=(30.0, 70.0),
        )

        # Should be able to calculate 50% improvement
        improvement_pct = ((result.evolved_mean / result.baseline_mean) - 1) * 100
        assert abs(improvement_pct - 50.0) < 0.1
