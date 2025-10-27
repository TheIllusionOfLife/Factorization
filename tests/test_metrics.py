"""Tests for fitness instrumentation and metrics tracking."""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from prototype import (
    SMALL_PRIMES,
    EvaluationMetrics,
    EvolutionaryEngine,
    FactorizationCrucible,
    Strategy,
)


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating EvaluationMetrics with all fields."""
        metrics = EvaluationMetrics(
            candidate_count=42,
            smoothness_scores=[1.5, 2.0, 1.8],
            timing_breakdown={
                "candidate_generation": 0.01,
                "modulus_filtering": 0.02,
                "smoothness_check": 0.03,
            },
            rejection_stats={"modulus_filter": 100, "min_hits": 50, "passed": 42},
            example_candidates=[12345, 67890],
        )

        assert metrics.candidate_count == 42
        assert len(metrics.smoothness_scores) == 3
        assert len(metrics.timing_breakdown) == 3
        assert metrics.rejection_stats["passed"] == 42
        assert len(metrics.example_candidates) == 2

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary for JSON export."""
        metrics = EvaluationMetrics(
            candidate_count=10,
            smoothness_scores=[1.2, 1.5],
            timing_breakdown={"test": 0.1},
            rejection_stats={"passed": 10},
            example_candidates=[100, 200],
        )

        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["candidate_count"] == 10
        assert "smoothness_scores" in metrics_dict
        assert "timing_breakdown" in metrics_dict


class TestDetailedEvaluation:
    """Test detailed strategy evaluation."""

    def test_detailed_evaluation_returns_metrics(self):
        """Test that detailed evaluation returns EvaluationMetrics."""
        crucible = FactorizationCrucible(961730063)
        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )

        metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.05)

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.candidate_count >= 0
        assert isinstance(metrics.smoothness_scores, list)
        assert isinstance(metrics.timing_breakdown, dict)
        assert isinstance(metrics.rejection_stats, dict)
        assert isinstance(metrics.example_candidates, list)

    def test_timing_breakdown_has_all_components(self):
        """Test that timing breakdown includes all expected components."""
        crucible = FactorizationCrucible(961730063)
        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )

        metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.05)

        assert "candidate_generation" in metrics.timing_breakdown
        assert "modulus_filtering" in metrics.timing_breakdown
        assert "smoothness_check" in metrics.timing_breakdown

        # All timing values should be non-negative
        for timing_value in metrics.timing_breakdown.values():
            assert timing_value >= 0

    def test_rejection_stats_tracked(self):
        """Test that rejection statistics are tracked correctly."""
        crucible = FactorizationCrucible(961730063)
        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )

        metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.05)

        assert "modulus_filter" in metrics.rejection_stats
        assert "min_hits" in metrics.rejection_stats
        assert "passed" in metrics.rejection_stats

        # Passed count should match candidate count
        assert metrics.rejection_stats["passed"] == metrics.candidate_count

    def test_smoothness_scores_limited(self):
        """Test that smoothness scores are limited to first 10 candidates."""
        crucible = FactorizationCrucible(961730063)
        # Very permissive strategy to find many candidates
        strategy = Strategy(
            power=2, modulus_filters=[(2, [0])], smoothness_bound=31, min_small_prime_hits=1
        )

        metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.1)

        # Should keep at most 10 smoothness scores
        assert len(metrics.smoothness_scores) <= 10

    def test_example_candidates_limited(self):
        """Test that example candidates are limited to first 5."""
        crucible = FactorizationCrucible(961730063)
        strategy = Strategy(
            power=2, modulus_filters=[(2, [0])], smoothness_bound=31, min_small_prime_hits=1
        )

        metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.1)

        # Should keep at most 5 example candidates
        assert len(metrics.example_candidates) <= 5

    def test_smoothness_score_calculation(self):
        """Test that smoothness scores are calculated correctly."""
        crucible = FactorizationCrucible(961730063)
        strategy = Strategy(
            power=2, modulus_filters=[(2, [0])], smoothness_bound=7, min_small_prime_hits=1
        )

        metrics = crucible.evaluate_strategy_detailed(strategy, duration_seconds=0.05)

        # Smoothness scores should be positive finite numbers
        for score in metrics.smoothness_scores:
            assert score > 0
            assert math.isfinite(score)


class TestMetricsHistoryTracking:
    """Test metrics history tracking in EvolutionaryEngine."""

    def test_metrics_history_initialized(self):
        """Test that metrics history is initialized."""
        crucible = FactorizationCrucible(961730063)
        engine = EvolutionaryEngine(
            crucible, population_size=3, llm_provider=None, evaluation_duration=0.05
        )

        assert hasattr(engine, "metrics_history")
        assert isinstance(engine.metrics_history, list)
        assert len(engine.metrics_history) == 0

    def test_metrics_tracked_per_generation(self):
        """Test that metrics are tracked for each generation."""
        crucible = FactorizationCrucible(961730063)
        engine = EvolutionaryEngine(
            crucible, population_size=3, llm_provider=None, evaluation_duration=0.05
        )

        engine.initialize_population()
        engine.run_evolutionary_cycle()

        # Should have metrics for one generation
        assert len(engine.metrics_history) == 1

        # Each generation should have metrics for each civilization
        generation_metrics = engine.metrics_history[0]
        assert len(generation_metrics) == 3  # population_size

        # Each entry should be EvaluationMetrics
        for metrics in generation_metrics:
            assert isinstance(metrics, EvaluationMetrics)

    def test_metrics_accumulated_across_generations(self):
        """Test that metrics accumulate across multiple generations."""
        crucible = FactorizationCrucible(961730063)
        engine = EvolutionaryEngine(
            crucible, population_size=3, llm_provider=None, evaluation_duration=0.05
        )

        engine.initialize_population()

        # Run 3 generations
        for _ in range(3):
            engine.run_evolutionary_cycle()

        # Should have metrics for 3 generations
        assert len(engine.metrics_history) == 3

        # Each generation should have metrics for population
        for generation_metrics in engine.metrics_history:
            assert len(generation_metrics) == 3


class TestMetricsExport:
    """Test metrics export to JSON."""

    def test_export_metrics_to_json(self):
        """Test exporting metrics history to JSON file."""
        crucible = FactorizationCrucible(961730063)
        engine = EvolutionaryEngine(
            crucible, population_size=2, llm_provider=None, evaluation_duration=0.05
        )

        engine.initialize_population()
        engine.run_evolutionary_cycle()

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            engine.export_metrics(temp_path)

            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)

            with open(temp_path) as f:
                data = json.load(f)

            # Verify structure
            assert "generation_count" in data
            assert "population_size" in data
            assert "metrics_history" in data
            assert "target_number" in data

            assert data["generation_count"] == 1
            assert data["population_size"] == 2
            assert len(data["metrics_history"]) == 1

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_includes_all_metric_fields(self):
        """Test that exported JSON includes all metric fields."""
        crucible = FactorizationCrucible(961730063)
        engine = EvolutionaryEngine(
            crucible, population_size=2, llm_provider=None, evaluation_duration=0.05
        )

        engine.initialize_population()
        engine.run_evolutionary_cycle()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            engine.export_metrics(temp_path)

            with open(temp_path) as f:
                data = json.load(f)

            # Check first civilization's metrics
            first_metrics = data["metrics_history"][0][0]

            assert "candidate_count" in first_metrics
            assert "smoothness_scores" in first_metrics
            assert "timing_breakdown" in first_metrics
            assert "rejection_stats" in first_metrics
            assert "example_candidates" in first_metrics

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIntegrationWithExistingCode:
    """Test integration with existing evolution code."""

    def test_backward_compatibility_evaluate_strategy(self):
        """Test that old evaluate_strategy method still works."""
        crucible = FactorizationCrucible(961730063)
        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )

        # Old method should still return integer
        fitness = crucible.evaluate_strategy(strategy, duration_seconds=0.05)
        assert isinstance(fitness, int)
        assert fitness >= 0

    def test_evolution_cycle_with_detailed_metrics(self):
        """Test full evolution cycle with detailed metrics enabled."""
        crucible = FactorizationCrucible(961730063)
        engine = EvolutionaryEngine(
            crucible, population_size=3, llm_provider=None, evaluation_duration=0.05
        )

        engine.initialize_population()

        # Run first generation evaluation
        engine.run_evolutionary_cycle()

        # Verify metrics were collected for generation 0
        assert len(engine.metrics_history) == 1
        assert all(isinstance(m, EvaluationMetrics) for m in engine.metrics_history[0])

        # Run second generation to verify metrics are stored for evaluated civs
        engine.run_evolutionary_cycle()

        # Verify metrics for generation 1
        assert len(engine.metrics_history) == 2
        assert all(isinstance(m, EvaluationMetrics) for m in engine.metrics_history[1])
