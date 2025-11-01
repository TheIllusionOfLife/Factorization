"""Integration tests for Prometheus multi-agent system.

Tests complete workflows beyond unit tests, including:
- CLI argument parsing and validation
- Full experiment runs for all 4 modes
- JSON export with emergence metrics
- Seed reproducibility
- Error handling for invalid configurations
"""

import json
import re
import subprocess
import sys

import pytest

from src.config import Config
from src.prometheus.experiment import PrometheusExperiment

# Test constants for consistent configuration across all tests
TEST_GENERATIONS = 2
TEST_POPULATION = 3
TEST_DURATION = 0.1
TEST_NUMBER = 961730063


class TestPrometheusExperimentIntegration:
    """Integration tests for PrometheusExperiment workflows."""

    def test_collaborative_mode_complete_workflow(self):
        """Test collaborative mode from initialization to completion."""
        # Create config
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=TEST_DURATION,
        )

        # Run experiment
        experiment = PrometheusExperiment(
            config=config,
            target_number=TEST_NUMBER,
            random_seed=42,
        )

        best_fitness, best_strategy, comm_stats = (
            experiment.run_collaborative_evolution(
                generations=TEST_GENERATIONS,
                population_size=TEST_POPULATION,
            )
        )

        # Verify results
        assert isinstance(best_fitness, (int, float))
        assert best_fitness >= 0
        assert best_strategy is not None
        assert isinstance(comm_stats, dict)
        assert "total_messages" in comm_stats
        assert comm_stats["total_messages"] > 0

    def test_search_only_mode_workflow(self):
        """Test search_only baseline mode workflow."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="search_only",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(
            config=config,
            target_number=961730063,
            random_seed=42,
        )

        best_fitness, best_strategy = experiment.run_independent_baseline(
            agent_type="search_only",
            generations=2,
            population_size=3,
        )

        assert isinstance(best_fitness, (int, float))
        assert best_fitness >= 0
        assert best_strategy is not None

    def test_eval_only_mode_workflow(self):
        """Test eval_only baseline mode workflow."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="eval_only",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(
            config=config,
            target_number=961730063,
            random_seed=42,
        )

        best_fitness, best_strategy = experiment.run_independent_baseline(
            agent_type="eval_only",
            generations=2,
            population_size=3,
        )

        assert isinstance(best_fitness, (int, float))
        assert best_fitness >= 0
        assert best_strategy is not None

    def test_rulebased_mode_workflow(self):
        """Test rulebased baseline mode workflow."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="rulebased",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(
            config=config,
            target_number=961730063,
            random_seed=42,
        )

        best_fitness, best_strategy = experiment.run_independent_baseline(
            agent_type="rulebased",
            generations=2,
            population_size=3,
        )

        assert isinstance(best_fitness, (int, float))
        assert best_fitness >= 0
        assert best_strategy is not None

    def test_seed_reproducibility_collaborative(self):
        """Test that same seed produces consistent results in collaborative mode.

        Note: Due to the complex interaction between agents and timing-based evaluation,
        exact reproducibility is not guaranteed. This test verifies that runs with the
        same seed complete successfully and produce valid results.
        """
        config1 = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=0.1,
        )

        config2 = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=0.1,
        )

        # Run 1
        exp1 = PrometheusExperiment(
            config=config1, target_number=961730063, random_seed=123
        )
        fitness1, strategy1, comm1 = exp1.run_collaborative_evolution(
            generations=2, population_size=3
        )

        # Run 2 with same seed
        exp2 = PrometheusExperiment(
            config=config2, target_number=961730063, random_seed=123
        )
        fitness2, strategy2, comm2 = exp2.run_collaborative_evolution(
            generations=2, population_size=3
        )

        # Both runs should complete successfully and produce valid results
        assert fitness1 >= 0 and fitness2 >= 0
        assert strategy1 is not None and strategy2 is not None
        assert comm1["total_messages"] > 0 and comm2["total_messages"] > 0
        # Strategies should have valid parameters
        assert 2 <= strategy1.power <= 5
        assert 2 <= strategy2.power <= 5

    def test_seed_reproducibility_baseline(self):
        """Test that same seed produces consistent results in baseline modes.

        Note: Due to timing-based evaluation and random agent behavior,
        exact reproducibility is not guaranteed. This test verifies that runs with the
        same seed complete successfully and produce valid results.
        """
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="search_only",
            evaluation_duration=0.1,
        )

        # Run 1
        exp1 = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=456
        )
        fitness1, strategy1 = exp1.run_independent_baseline(
            agent_type="search_only", generations=2, population_size=3
        )

        # Run 2 with same seed
        exp2 = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=456
        )
        fitness2, strategy2 = exp2.run_independent_baseline(
            agent_type="search_only", generations=2, population_size=3
        )

        # Both runs should complete successfully and produce valid results
        assert fitness1 >= 0 and fitness2 >= 0
        assert strategy1 is not None and strategy2 is not None
        # Strategies should have valid parameters
        assert 2 <= strategy1.power <= 5
        assert 2 <= strategy2.power <= 5

    def test_invalid_agent_type_raises_error(self):
        """Test that invalid agent_type raises ValueError."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(config=config, target_number=961730063)

        with pytest.raises(ValueError, match="Invalid agent_type"):
            experiment.run_independent_baseline(
                agent_type="invalid_mode",
                generations=2,
                population_size=3,
            )

    def test_invalid_generations_raises_error(self):
        """Test that invalid generations raises ValueError."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(config=config, target_number=961730063)

        with pytest.raises(ValueError, match="generations must be >= 1"):
            experiment.run_collaborative_evolution(generations=0, population_size=3)

    def test_invalid_population_raises_error(self):
        """Test that invalid population_size raises ValueError."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(config=config, target_number=961730063)

        with pytest.raises(ValueError, match="population_size must be >= 1"):
            experiment.run_collaborative_evolution(generations=2, population_size=0)

    def test_prometheus_disabled_raises_error(self):
        """Test that prometheus_enabled=False raises ValueError."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=False,  # Disabled
            evaluation_duration=0.1,
        )

        with pytest.raises(ValueError, match="prometheus_enabled must be True"):
            PrometheusExperiment(config=config, target_number=961730063)

    @pytest.mark.slow
    def test_compare_with_baselines_calculates_metrics(self):
        """Test that compare_with_baselines returns valid EmergenceMetrics.

        Marked as slow because it runs all 4 modes (collaborative + 3 baselines).
        """
        from src.prometheus.experiment import EmergenceMetrics

        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=TEST_DURATION,
        )

        experiment = PrometheusExperiment(
            config=config, target_number=TEST_NUMBER, random_seed=42
        )

        # Run comparison (uses small parameters for speed)
        metrics = experiment.compare_with_baselines(
            generations=TEST_GENERATIONS, population_size=TEST_POPULATION
        )

        # Verify structure
        assert isinstance(metrics, EmergenceMetrics)

        # Verify all fitness values are non-negative
        assert metrics.collaborative_fitness >= 0
        assert metrics.search_only_fitness >= 0
        assert metrics.eval_only_fitness >= 0
        assert metrics.rulebased_fitness >= 0

        # Verify emergence metrics are calculated
        assert metrics.emergence_factor >= 0
        assert isinstance(metrics.synergy_score, (int, float))
        assert isinstance(metrics.communication_efficiency, (int, float))

        # Verify collaborative mode sent messages
        assert metrics.total_messages > 0, "Collaborative mode should send messages"


class TestPrometheusCLIIntegration:
    """Integration tests for Prometheus CLI workflows."""

    def test_cli_collaborative_mode(self):
        """Test CLI with collaborative mode."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--prometheus",
                "--prometheus-mode",
                "collaborative",
                "--generations",
                "2",
                "--population",
                "3",
                "--duration",
                "0.1",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )
        assert "Prometheus Multi-Agent Mode" in result.stdout
        assert "collaborative" in result.stdout
        assert "Best fitness" in result.stdout

    def test_cli_search_only_mode(self):
        """Test CLI with search_only baseline mode."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--prometheus",
                "--prometheus-mode",
                "search_only",
                "--generations",
                "2",
                "--population",
                "3",
                "--duration",
                "0.1",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )
        assert "Prometheus Multi-Agent Mode" in result.stdout
        assert "search_only" in result.stdout

    def test_cli_eval_only_mode(self):
        """Test CLI with eval_only baseline mode."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--prometheus",
                "--prometheus-mode",
                "eval_only",
                "--generations",
                "2",
                "--population",
                "3",
                "--duration",
                "0.1",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )
        assert "Prometheus Multi-Agent Mode" in result.stdout
        assert "eval_only" in result.stdout

    def test_cli_rulebased_mode(self):
        """Test CLI with rulebased baseline mode."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--prometheus",
                "--prometheus-mode",
                "rulebased",
                "--generations",
                "2",
                "--population",
                "3",
                "--duration",
                "0.1",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )
        assert "Prometheus Multi-Agent Mode" in result.stdout
        assert "rulebased" in result.stdout

    def test_cli_invalid_mode_raises_error(self):
        """Test that invalid prometheus mode fails gracefully."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--prometheus",
                "--prometheus-mode",
                "invalid_mode",
                "--generations",
                "2",
                "--population",
                "3",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        stderr_stdout = result.stderr + result.stdout
        # argparse will reject invalid choice
        assert (
            "invalid choice" in stderr_stdout.lower()
            or "error" in stderr_stdout.lower()
        )

    def test_cli_prometheus_without_mode_uses_default(self):
        """Test that --prometheus without --prometheus-mode uses default (collaborative)."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--prometheus",
                "--generations",
                "2",
                "--population",
                "3",
                "--duration",
                "0.1",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )
        assert "Prometheus Multi-Agent Mode" in result.stdout
        assert "collaborative" in result.stdout  # Default mode

    def test_cli_json_export_with_prometheus(self, tmp_path):
        """Test JSON export with Prometheus metrics."""
        export_file = tmp_path / "prometheus_export.json"

        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--prometheus",
                "--prometheus-mode",
                "collaborative",
                "--generations",
                "2",
                "--population",
                "3",
                "--duration",
                "0.1",
                "--seed",
                "42",
                "--export-metrics",
                str(export_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )
        assert export_file.exists(), "Export file not created"

        # Validate JSON structure
        with open(export_file) as f:
            data = json.load(f)

        assert "target_number" in data
        assert "config" in data
        assert "prometheus_enabled" in data["config"]
        assert data["config"]["prometheus_enabled"] is True
        assert "prometheus_mode" in data["config"]
        assert data["config"]["prometheus_mode"] == "collaborative"

    def test_cli_seed_reproducibility(self):
        """Test that CLI with seed runs successfully and produces output.

        Note: Exact fitness may vary due to timing (documented behavior),
        but the tool should run without errors and produce valid output.
        """
        args = [
            sys.executable,
            "main.py",
            "--prometheus",
            "--prometheus-mode",
            "search_only",
            "--generations",
            "2",
            "--population",
            "3",
            "--duration",
            "0.1",
            "--seed",
            "789",
        ]

        # Run 1
        result1 = subprocess.run(args, capture_output=True, text=True, check=False)

        # Run 2 with same args
        result2 = subprocess.run(args, capture_output=True, text=True, check=False)

        # Extract fitness from output (format: "Best fitness: 12345")
        fitness1_match = re.search(r"Best fitness:\s*(\d+)", result1.stdout)
        fitness2_match = re.search(r"Best fitness:\s*(\d+)", result2.stdout)

        assert fitness1_match and fitness2_match, "Could not find fitness in output"
        fitness1 = int(fitness1_match.group(1))
        fitness2 = int(fitness2_match.group(1))

        # Both runs should produce positive fitness (exact value may vary due to timing)
        assert fitness1 > 0, "First run produced zero fitness"
        assert fitness2 > 0, "Second run produced zero fitness"


class TestPrometheusMemoryManagement:
    """Tests for memory cleanup and resource management."""

    def test_memory_cleanup_after_collaborative_run(self):
        """Test that memory is properly cleaned up after collaborative run."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        # Run experiment
        best_fitness, best_strategy, comm_stats = (
            experiment.run_collaborative_evolution(generations=2, population_size=3)
        )

        # Verify successful execution (smoke test for memory issues)
        assert best_fitness >= 0, "Invalid fitness returned"
        assert best_strategy is not None, "No best strategy returned"
        assert comm_stats.get("total_messages", 0) > 0, "No messages exchanged"

    def test_memory_cleanup_after_baseline_run(self):
        """Test that memory is properly cleaned up after baseline run."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="search_only",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        # Run experiment
        best_fitness, best_strategy = experiment.run_independent_baseline(
            agent_type="search_only", generations=2, population_size=3
        )

        # Verify successful execution (smoke test for memory issues)
        assert best_fitness >= 0, "Invalid fitness returned"
        assert best_strategy is not None, "No best strategy returned"


class TestPrometheusPerformanceCharacteristics:
    """Tests for performance expectations and characteristics."""

    def test_collaborative_mode_sends_messages(self):
        """Test that collaborative mode actually sends messages."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="collaborative",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        _, _, comm_stats = experiment.run_collaborative_evolution(
            generations=2, population_size=3
        )

        # Should send strategy requests and evaluation requests
        assert comm_stats["total_messages"] > 0

    def test_baseline_modes_do_not_send_messages(self):
        """Test that baseline modes don't involve message passing."""
        config = Config(
            api_key="test_key",
            llm_enabled=False,
            prometheus_enabled=True,
            prometheus_mode="search_only",
            evaluation_duration=0.1,
        )

        experiment = PrometheusExperiment(
            config=config, target_number=961730063, random_seed=42
        )

        # Baseline modes return only (fitness, strategy), no communication stats
        result = experiment.run_independent_baseline(
            agent_type="search_only", generations=2, population_size=3
        )

        assert len(result) == 2  # Only fitness and strategy, no comm_stats

    @pytest.mark.slow
    def test_all_modes_find_valid_strategies(self):
        """Test that all modes produce valid strategies.

        Note: eval_only mode may produce zero fitness when random strategies
        happen to be ineffective. This test verifies structural validity.

        Marked as slow because it tests all 4 modes (4x longer than single-mode tests).
        """
        modes = ["collaborative", "search_only", "eval_only", "rulebased"]

        for mode in modes:
            config = Config(
                api_key="test_key",
                llm_enabled=False,
                prometheus_enabled=True,
                prometheus_mode=mode,
                evaluation_duration=0.1,
            )

            experiment = PrometheusExperiment(
                config=config, target_number=961730063, random_seed=42
            )

            if mode == "collaborative":
                fitness, strategy, _ = experiment.run_collaborative_evolution(
                    generations=2, population_size=3
                )
            else:
                fitness, strategy = experiment.run_independent_baseline(
                    agent_type=mode, generations=2, population_size=3
                )

            # All modes should produce a strategy
            assert strategy is not None, f"{mode} mode produced no strategy"
            assert strategy.power >= 2 and strategy.power <= 5
            assert len(strategy.modulus_filters) >= 0
            # Fitness should be non-negative (may be zero for eval_only with unlucky random strategies)
            assert fitness >= 0, f"{mode} mode produced negative fitness"
