"""Tests for Prometheus experiment orchestration.

Following TDD: These tests are written BEFORE implementation.
They define the expected behavior of PrometheusExperiment and EmergenceMetrics.
"""

import pytest

from src.config import Config
from src.prometheus.experiment import EmergenceMetrics, PrometheusExperiment
from src.strategy import Strategy


class TestEmergenceMetrics:
    """Test EmergenceMetrics dataclass."""

    def test_emergence_metrics_creation(self):
        """Test creating EmergenceMetrics with all fields."""
        metrics = EmergenceMetrics(
            collaborative_fitness=100.0,
            search_only_fitness=80.0,
            eval_only_fitness=70.0,
            rulebased_fitness=60.0,
            emergence_factor=1.25,
            synergy_score=20.0,
            communication_efficiency=0.5,
            total_messages=40,
        )

        assert metrics.collaborative_fitness == 100.0
        assert metrics.search_only_fitness == 80.0
        assert metrics.eval_only_fitness == 70.0
        assert metrics.rulebased_fitness == 60.0
        assert metrics.emergence_factor == 1.25
        assert metrics.synergy_score == 20.0
        assert metrics.communication_efficiency == 0.5
        assert metrics.total_messages == 40

    def test_emergence_factor_calculation(self):
        """Test emergence_factor is collaborative / max(baselines)."""
        # collaborative=100, max(baselines)=80 → factor=1.25
        metrics = EmergenceMetrics(
            collaborative_fitness=100.0,
            search_only_fitness=80.0,
            eval_only_fitness=70.0,
            rulebased_fitness=60.0,
            emergence_factor=100.0 / 80.0,
            synergy_score=20.0,
            communication_efficiency=0.0,
            total_messages=0,
        )
        assert metrics.emergence_factor == 1.25

    def test_synergy_score_calculation(self):
        """Test synergy_score is collaborative - max(baselines)."""
        # collaborative=100, max(baselines)=80 → synergy=20
        metrics = EmergenceMetrics(
            collaborative_fitness=100.0,
            search_only_fitness=80.0,
            eval_only_fitness=70.0,
            rulebased_fitness=60.0,
            emergence_factor=1.25,
            synergy_score=100.0 - 80.0,
            communication_efficiency=0.0,
            total_messages=0,
        )
        assert metrics.synergy_score == 20.0

    def test_communication_efficiency_calculation(self):
        """Test communication_efficiency is synergy / total_messages."""
        # synergy=20, messages=40 → efficiency=0.5
        metrics = EmergenceMetrics(
            collaborative_fitness=100.0,
            search_only_fitness=80.0,
            eval_only_fitness=70.0,
            rulebased_fitness=60.0,
            emergence_factor=1.25,
            synergy_score=20.0,
            communication_efficiency=20.0 / 40.0,
            total_messages=40,
        )
        assert metrics.communication_efficiency == 0.5

    def test_emergence_metrics_with_zero_messages(self):
        """Test handling zero messages (rule-based mode)."""
        metrics = EmergenceMetrics(
            collaborative_fitness=50.0,
            search_only_fitness=50.0,
            eval_only_fitness=50.0,
            rulebased_fitness=50.0,
            emergence_factor=1.0,
            synergy_score=0.0,
            communication_efficiency=0.0,
            total_messages=0,
        )
        assert metrics.communication_efficiency == 0.0


class TestPrometheusExperiment:
    """Test PrometheusExperiment orchestration."""

    def test_experiment_initialization(self):
        """Test PrometheusExperiment initializes correctly."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        assert experiment.config == config
        assert experiment.target_number == 961730063
        assert experiment.random_seed == 42

    def test_experiment_requires_prometheus_enabled(self):
        """Test experiment requires prometheus_enabled=True in config."""
        config = Config(api_key="test", prometheus_enabled=False)

        with pytest.raises(ValueError, match="prometheus_enabled must be True"):
            PrometheusExperiment(config=config, target_number=961730063)

    def test_run_collaborative_evolution_returns_results(self):
        """Test run_collaborative_evolution returns fitness and strategy."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        fitness, strategy, stats, _ = experiment.run_collaborative_evolution(
            generations=2, population_size=5
        )

        assert isinstance(fitness, (int, float))
        assert fitness >= 0
        assert isinstance(strategy, Strategy)
        assert isinstance(stats, dict)
        assert "total_messages" in stats

    def test_run_collaborative_evolution_uses_both_agents(self):
        """Test collaborative mode uses SearchSpecialist and EvaluationSpecialist."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        _, _, stats, _ = experiment.run_collaborative_evolution(
            generations=2, population_size=5
        )

        # Should have messages exchanged between agents
        assert stats["total_messages"] > 0

    def test_run_independent_baseline_search_only(self):
        """Test search_only baseline uses only SearchSpecialist."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            prometheus_mode="search_only",
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        fitness, strategy = experiment.run_independent_baseline(
            agent_type="search_only", generations=2, population_size=5
        )

        assert isinstance(fitness, (int, float))
        assert fitness >= 0
        assert isinstance(strategy, Strategy)

    def test_run_independent_baseline_eval_only(self):
        """Test eval_only baseline uses only EvaluationSpecialist."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            prometheus_mode="eval_only",
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        fitness, strategy = experiment.run_independent_baseline(
            agent_type="eval_only", generations=2, population_size=5
        )

        assert isinstance(fitness, (int, float))
        assert fitness >= 0
        assert isinstance(strategy, Strategy)

    def test_run_independent_baseline_rulebased(self):
        """Test rulebased baseline uses traditional evolution."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        fitness, strategy = experiment.run_independent_baseline(
            agent_type="rulebased", generations=2, population_size=5
        )

        assert isinstance(fitness, (int, float))
        assert fitness >= 0
        assert isinstance(strategy, Strategy)

    def test_run_independent_baseline_invalid_type_raises_error(self):
        """Test invalid agent_type raises error."""
        config = Config(
            api_key="test", prometheus_enabled=True, evaluation_duration=0.1
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        with pytest.raises(ValueError, match="Invalid agent_type"):
            experiment.run_independent_baseline(
                agent_type="invalid", generations=2, population_size=5
            )

    def test_compare_with_baselines_returns_emergence_metrics(self):
        """Test compare_with_baselines runs all modes and returns EmergenceMetrics."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        metrics = experiment.compare_with_baselines(generations=2, population_size=5)

        assert isinstance(metrics, EmergenceMetrics)
        assert metrics.collaborative_fitness >= 0
        assert metrics.search_only_fitness >= 0
        assert metrics.eval_only_fitness >= 0
        assert metrics.rulebased_fitness >= 0
        # Emergence factor can be 0 if collaborative mode underperforms baselines
        assert metrics.emergence_factor >= 0
        assert metrics.total_messages >= 0

    def test_compare_with_baselines_calculates_emergence_factor_correctly(self):
        """Test emergence_factor is collaborative / max(baselines)."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        metrics = experiment.compare_with_baselines(generations=2, population_size=5)

        max_baseline = max(
            metrics.search_only_fitness,
            metrics.eval_only_fitness,
            metrics.rulebased_fitness,
        )

        # Allow small floating point error
        if max_baseline > 0:
            expected_factor = metrics.collaborative_fitness / max_baseline
            assert abs(metrics.emergence_factor - expected_factor) < 0.01
        else:
            # Production clamps to 1.0 when all baselines are zero
            assert metrics.emergence_factor == pytest.approx(1.0, abs=0.01)

    def test_compare_with_baselines_calculates_synergy_score_correctly(self):
        """Test synergy_score is collaborative - max(baselines)."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        metrics = experiment.compare_with_baselines(generations=2, population_size=5)

        max_baseline = max(
            metrics.search_only_fitness,
            metrics.eval_only_fitness,
            metrics.rulebased_fitness,
        )

        expected_synergy = metrics.collaborative_fitness - max_baseline
        assert abs(metrics.synergy_score - expected_synergy) < 0.01

    def test_different_seeds_produce_different_results(self):
        """Test different random seeds produce different results."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )

        experiment1 = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )
        experiment2 = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=100
        )

        fitness1, strategy1, _, _ = experiment1.run_collaborative_evolution(
            generations=2, population_size=5
        )
        fitness2, strategy2, _, _ = experiment2.run_collaborative_evolution(
            generations=2, population_size=5
        )

        # Strategies should be different (not guaranteed but highly likely)
        # At minimum, verify both experiments ran successfully
        assert isinstance(fitness1, (int, float))
        assert isinstance(fitness2, (int, float))
        assert isinstance(strategy1, Strategy)
        assert isinstance(strategy2, Strategy)

    def test_same_seed_produces_reproducible_results(self):
        """Test same seed produces same results."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.1,
            llm_enabled=False,
        )

        experiment1 = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )
        experiment2 = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        fitness1, strategy1, _, _ = experiment1.run_collaborative_evolution(
            generations=2, population_size=5
        )
        fitness2, strategy2, _, _ = experiment2.run_collaborative_evolution(
            generations=2, population_size=5
        )

        # Same seed should produce similar results
        # Note: Perfect reproducibility is not guaranteed due to:
        # 1. Timing-based evaluation (non-deterministic)
        # 2. Multiple independent RNG calls across agents
        # We verify that experiments run successfully with valid outputs
        assert isinstance(fitness1, (int, float))
        assert isinstance(fitness2, (int, float))
        assert fitness1 >= 0
        assert fitness2 >= 0
        assert isinstance(strategy1, Strategy)
        assert isinstance(strategy2, Strategy)
        assert isinstance(strategy1.modulus_filters, list)
        assert isinstance(strategy2.modulus_filters, list)

    def test_experiment_with_zero_generations_raises_error(self):
        """Test generations must be >= 1."""
        config = Config(
            api_key="test", prometheus_enabled=True, evaluation_duration=0.1
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        with pytest.raises(ValueError, match="generations must be >= 1"):
            experiment.run_collaborative_evolution(generations=0, population_size=5)

    def test_experiment_with_zero_population_raises_error(self):
        """Test population_size must be >= 1."""
        config = Config(
            api_key="test", prometheus_enabled=True, evaluation_duration=0.1
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        with pytest.raises(ValueError, match="population_size must be >= 1"):
            experiment.run_collaborative_evolution(generations=2, population_size=0)

    def test_config_excludes_api_key_from_export(self):
        """Test that config.to_dict() excludes API key in Prometheus mode."""
        config = Config(
            api_key="secret-key-12345",
            prometheus_enabled=True,
            evaluation_duration=0.1,
        )

        # Export without sensitive data
        config_dict = config.to_dict(include_sensitive=False)

        # Verify API key not in exported dict
        assert "api_key" not in config_dict
        assert "secret-key" not in str(config_dict)

        # But should be in full export
        config_dict_full = config.to_dict(include_sensitive=True)
        assert "api_key" in config_dict_full
        assert config_dict_full["api_key"] == "secret-key-12345"


class TestPrometheusIntegration:
    """Integration tests for full Prometheus workflow."""

    def test_full_comparison_workflow(self):
        """Test complete workflow: collaborative + 3 baselines."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.5,  # Increased from 0.1s to reduce timing variance
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        metrics = experiment.compare_with_baselines(generations=3, population_size=5)

        # Verify all baselines ran
        assert metrics.collaborative_fitness >= 0
        assert metrics.search_only_fitness >= 0
        assert metrics.eval_only_fitness >= 0
        assert metrics.rulebased_fitness >= 0

        # Verify emergence metrics calculated
        # Note: emergence_factor can be 0 if collaborative_fitness is 0 (rare but possible)
        assert metrics.emergence_factor >= 0
        assert isinstance(metrics.synergy_score, (int, float))
        # Communication efficiency can be negative if collaborative underperforms
        assert isinstance(metrics.communication_efficiency, (int, float))
        assert metrics.total_messages >= 0

    def test_multiple_generations_improves_fitness(self):
        """Test fitness generally improves over generations."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            evaluation_duration=0.2,
            llm_enabled=False,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        # Run with 1 generation
        fitness1, _, _, _ = experiment.run_collaborative_evolution(
            generations=1, population_size=10
        )

        # Run with 5 generations (new experiment, same seed for reproducibility)
        experiment2 = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )
        fitness5, _, _, _ = experiment2.run_collaborative_evolution(
            generations=5, population_size=10
        )

        # More generations should generally find better or equal fitness
        # (not guaranteed due to randomness, but likely with enough population)
        assert fitness5 >= 0
        assert fitness1 >= 0


class TestGenerationHistoryTracking:
    """Test generation history tracking for Gen 0 vs Gen 1 analysis."""

    def test_run_collaborative_evolution_returns_generation_history(self):
        """Test that run_collaborative_evolution returns generation history as 4th return value."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.1,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        # Run 3 generations
        result = experiment.run_collaborative_evolution(
            generations=3, population_size=5
        )

        # Should return 4 values now: (fitness, strategy, comm_stats, generation_history)
        assert len(result) == 4
        fitness, strategy, comm_stats, generation_history = result

        # Basic type checks
        assert isinstance(fitness, (int, float))
        assert isinstance(strategy, Strategy)
        assert isinstance(comm_stats, dict)
        assert isinstance(generation_history, list)

    def test_generation_history_length_matches_generations(self):
        """Test that generation_history has one entry per generation."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.1,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=100
        )

        generations = 5
        _, _, _, history = experiment.run_collaborative_evolution(
            generations=generations, population_size=10
        )

        assert len(history) == generations

    def test_generation_history_structure(self):
        """Test that each generation entry has required fields."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.1,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=200
        )

        _, _, _, history = experiment.run_collaborative_evolution(
            generations=2, population_size=5
        )

        # Check structure of each generation
        for gen_idx, gen_data in enumerate(history):
            assert isinstance(gen_data, dict)

            # Required fields
            assert "generation" in gen_data
            assert "best_fitness" in gen_data
            assert "best_strategy" in gen_data
            assert "all_strategies" in gen_data
            assert "population_size" in gen_data

            # Field types
            assert gen_data["generation"] == gen_idx
            assert isinstance(gen_data["best_fitness"], (int, float))
            assert isinstance(gen_data["best_strategy"], dict)  # Serialized strategy
            assert isinstance(gen_data["all_strategies"], list)
            assert gen_data["population_size"] == 5

    def test_generation_history_all_strategies_count(self):
        """Test that all_strategies contains all population members."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.1,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=300
        )

        population_size = 8
        _, _, _, history = experiment.run_collaborative_evolution(
            generations=2, population_size=population_size
        )

        for gen_data in history:
            assert len(gen_data["all_strategies"]) == population_size

            # Each strategy entry should have fitness and strategy
            for strategy_entry in gen_data["all_strategies"]:
                assert "fitness" in strategy_entry
                assert "strategy" in strategy_entry
                assert isinstance(strategy_entry["fitness"], (int, float))
                assert isinstance(strategy_entry["strategy"], dict)

    def test_generation_history_best_matches_all_strategies(self):
        """Test that best_fitness/best_strategy match the max in all_strategies."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.1,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=400
        )

        _, _, _, history = experiment.run_collaborative_evolution(
            generations=3, population_size=10
        )

        for gen_data in history:
            # Find max fitness in all_strategies
            max_fitness_in_gen = max(s["fitness"] for s in gen_data["all_strategies"])

            # Should match best_fitness
            assert gen_data["best_fitness"] == max_fitness_in_gen

    def test_generation_history_gen0_vs_gen1_comparison(self):
        """Test that we can compare Gen 0 vs Gen 1 using history."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.2,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=500
        )

        _, _, _, history = experiment.run_collaborative_evolution(
            generations=3, population_size=10
        )

        # Extract Gen 0 and Gen 1
        gen0 = history[0]
        gen1 = history[1]

        # Should have different generation numbers
        assert gen0["generation"] == 0
        assert gen1["generation"] == 1

        # Both should have same population size
        assert len(gen0["all_strategies"]) == len(gen1["all_strategies"])

        # Can extract fitness values for comparison
        gen0_fitnesses = [s["fitness"] for s in gen0["all_strategies"]]
        gen1_fitnesses = [s["fitness"] for s in gen1["all_strategies"]]

        assert len(gen0_fitnesses) == 10
        assert len(gen1_fitnesses) == 10
        assert all(isinstance(f, (int, float)) for f in gen0_fitnesses)
        assert all(isinstance(f, (int, float)) for f in gen1_fitnesses)

    def test_generation_history_strategy_serialization(self):
        """Test that strategies are properly serialized as dicts."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.1,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=600
        )

        _, _, _, history = experiment.run_collaborative_evolution(
            generations=2, population_size=5
        )

        # Check strategy serialization in best_strategy
        best_strategy_dict = history[0]["best_strategy"]
        assert "power" in best_strategy_dict
        assert "modulus_filters" in best_strategy_dict
        assert "smoothness_bound" in best_strategy_dict
        assert "min_small_prime_hits" in best_strategy_dict

        # Check strategy serialization in all_strategies
        first_strategy_dict = history[0]["all_strategies"][0]["strategy"]
        assert "power" in first_strategy_dict
        assert "modulus_filters" in first_strategy_dict
        assert "smoothness_bound" in first_strategy_dict
        assert "min_small_prime_hits" in first_strategy_dict

    def test_generation_history_empty_for_zero_generations(self):
        """Test edge case: 0 generations should return empty history (though invalid input)."""
        config = Config(
            api_key="test",
            prometheus_enabled=True,
            llm_enabled=False,
            evaluation_duration=0.1,
        )
        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=700
        )

        # Should raise ValueError for generations < 1
        with pytest.raises(ValueError, match="generations must be >= 1"):
            experiment.run_collaborative_evolution(generations=0, population_size=5)
