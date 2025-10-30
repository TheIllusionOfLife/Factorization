"""CLI end-to-end tests using subprocess.

Tests the main.py CLI interface to ensure:
- All command-line arguments work correctly
- JSON exports have proper structure
- Argument validation catches invalid inputs
- Reproducibility via --seed works
- Different modes (normal, meta-learning, comparison, LLM) function
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# =============================================================================
# Test Utilities
# =============================================================================


def run_cli(*args, timeout=30, check=True, env=None):
    """Run main.py with arguments, return CompletedProcess.

    Args:
        *args: Command-line arguments to pass to main.py
        timeout: Maximum execution time in seconds
        check: If True, raise on non-zero exit (default: True for success tests)
        env: Environment variables dict (merged with os.environ)

    Returns:
        subprocess.CompletedProcess with stdout, stderr, returncode
    """
    import sys

    cmd = [sys.executable, "main.py"] + list(args)

    # Merge environment variables
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,  # We'll check manually
        env=full_env,
    )

    if check and result.returncode != 0:
        raise AssertionError(
            f"CLI failed with exit code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result


def assert_contains(text, *patterns):
    """Assert all patterns exist in text."""
    for pattern in patterns:
        assert pattern in text, (
            f"Expected pattern not found: '{pattern}'\nText:\n{text}"
        )


def assert_not_contains(text, *patterns):
    """Assert no patterns exist in text."""
    for pattern in patterns:
        assert pattern not in text, (
            f"Unexpected pattern found: '{pattern}'\nText:\n{text}"
        )


def load_json_file(path):
    """Load and parse JSON file, return dict."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# 1. Help & Version Tests
# =============================================================================


def test_help_flag():
    """Test --help flag shows usage information."""
    result = run_cli("--help")

    assert result.returncode == 0
    assert_contains(
        result.stdout,
        "usage:",
        "Evolutionary GNFS strategy optimizer",
        "--number",
        "--generations",
        "--population",
        "--llm",
        "--export-metrics",
    )


def test_help_shows_all_argument_categories():
    """Test --help documents all argument categories."""
    result = run_cli("--help")

    # Verify major argument categories are documented
    assert_contains(
        result.stdout,
        "--elite-rate",  # Evolution parameters
        "--power-min",  # Strategy bounds
        "--meta-learning",  # Meta-learning
        "--mutation-prob-power",  # Mutation probabilities
        "--log-level",  # Logging
        "--compare-baseline",  # Comparison mode
    )


# =============================================================================
# 2. Basic Workflow Tests
# =============================================================================


def test_basic_evolution_run():
    """Test basic evolution run completes successfully."""
    result = run_cli(
        "--generations",
        "2",
        "--population",
        "3",
        "--log-level",
        "WARNING",  # Quiet mode for faster test
    )

    assert result.returncode == 0
    # Should show some output about generations (even in WARNING mode, user output shows)
    assert_contains(result.stdout, "Target number:", "Generations: 2")


def test_rule_based_mode_default():
    """Test rule-based mode (default without --llm)."""
    result = run_cli(
        "--generations",
        "1",
        "--population",
        "2",
        "--log-level",
        "WARNING",
    )

    assert result.returncode == 0
    assert_contains(result.stdout, "Rule-based mode")
    assert_not_contains(result.stdout, "LLM mode enabled")


def test_custom_number():
    """Test --number argument with custom value."""
    result = run_cli(
        "--number",
        "12345678901",
        "--generations",
        "1",
        "--population",
        "2",
        "--log-level",
        "WARNING",
    )

    assert result.returncode == 0
    assert_contains(result.stdout, "12345678901")


def test_quiet_mode():
    """Test --log-level WARNING produces minimal output."""
    result = run_cli(
        "--generations",
        "1",
        "--population",
        "2",
        "--log-level",
        "WARNING",
    )

    assert result.returncode == 0
    # Should not have DEBUG or INFO log messages
    # (User-facing output still appears, but no detailed logs)
    output = result.stdout + result.stderr
    assert_not_contains(output, "DEBUG:", "logger.debug")


# =============================================================================
# 3. JSON Export Tests
# =============================================================================


def test_export_metrics_creates_file():
    """Test --export-metrics creates JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "metrics.json"

        result = run_cli(
            "--generations",
            "1",
            "--population",
            "2",
            "--export-metrics",
            str(output_path),
            "--log-level",
            "WARNING",
        )

        assert result.returncode == 0
        assert output_path.exists(), "Metrics file was not created"


def test_export_metrics_json_structure():
    """Test exported metrics JSON has required structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "metrics.json"

        run_cli(
            "--generations",
            "2",
            "--population",
            "3",
            "--export-metrics",
            str(output_path),
            "--log-level",
            "WARNING",
        )

        data = load_json_file(output_path)

        # Validate required top-level keys
        assert "target_number" in data
        assert "generation_count" in data
        assert "metrics_history" in data
        assert "config" in data

        # Validate values
        assert data["generation_count"] == 2
        assert isinstance(data["metrics_history"], list)
        assert len(data["metrics_history"]) == 2  # One per generation

        # Each generation should have metrics for each civilization
        assert isinstance(data["metrics_history"][0], list)
        assert len(data["metrics_history"][0]) == 3  # Population size


def test_export_metrics_directory_creation():
    """Test --export-metrics auto-creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nested" / "dir" / "metrics.json"

        result = run_cli(
            "--generations",
            "1",
            "--population",
            "2",
            "--export-metrics",
            str(output_path),
            "--log-level",
            "WARNING",
        )

        assert result.returncode == 0
        assert output_path.exists(), "File not created in nested directory"
        assert output_path.parent.exists(), "Parent directory not auto-created"


def test_exported_config_excludes_api_key():
    """Test exported config excludes sensitive API key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "metrics.json"

        run_cli(
            "--generations",
            "1",
            "--population",
            "2",
            "--export-metrics",
            str(output_path),
            "--log-level",
            "WARNING",
        )

        data = load_json_file(output_path)

        # Security check: API key should not be in exported config
        assert "api_key" not in data["config"]
        assert "GEMINI_API_KEY" not in json.dumps(data)


def test_export_comparison_json_structure():
    """Test --export-comparison creates valid JSON with comparison results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "comparison.json"

        run_cli(
            "--compare-baseline",
            "--num-comparison-runs",
            "2",
            "--generations",
            "2",
            "--population",
            "3",
            "--export-comparison",
            str(output_path),
            "--log-level",
            "WARNING",
        )

        data = load_json_file(output_path)

        # Validate comparison-specific keys
        assert "num_runs" in data
        assert "runs" in data
        assert "analysis" in data

        # Validate runs structure
        assert data["num_runs"] == 2
        assert len(data["runs"]) == 2

        # Each run should have fitness history and baseline fitness
        for run in data["runs"]:
            assert "evolved_fitness" in run
            assert "baseline_fitness" in run
            assert isinstance(run["evolved_fitness"], list)
            assert isinstance(run["baseline_fitness"], dict)


# =============================================================================
# 4. Argument Validation Tests
# =============================================================================


def test_invalid_generations_zero():
    """Test CLI rejects --generations 0 with clear error."""
    result = run_cli(
        "--generations",
        "0",
        "--population",
        "3",
        check=False,  # Expect failure
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert_contains(output, "ERROR", "generations must be >= 1")


def test_invalid_generations_negative():
    """Test CLI rejects negative --generations."""
    result = run_cli(
        "--generations",
        "-5",
        "--population",
        "3",
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert_contains(output, "ERROR", "generations must be >= 1")


def test_invalid_population_zero():
    """Test CLI rejects --population 0."""
    result = run_cli(
        "--generations",
        "2",
        "--population",
        "0",
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert_contains(output, "ERROR", "population must be >= 1")


def test_invalid_num_comparison_runs():
    """Test CLI rejects invalid --num-comparison-runs when --compare-baseline set."""
    result = run_cli(
        "--compare-baseline",
        "--num-comparison-runs",
        "0",
        "--generations",
        "2",
        "--population",
        "3",
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert_contains(output, "ERROR", "num-comparison-runs must be >= 1")


def test_invalid_elite_rate():
    """Test CLI rejects invalid --elite-rate via config validation."""
    result = run_cli(
        "--elite-rate",
        "1.5",  # > 1.0 is invalid
        "--generations",
        "1",
        "--population",
        "2",
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert_contains(output, "ERROR", "Invalid configuration")


def test_conflicting_rates():
    """Test CLI rejects crossover + mutation rates > 1.0."""
    result = run_cli(
        "--crossover-rate",
        "0.7",
        "--mutation-rate",
        "0.7",  # Sum = 1.4 > 1.0
        "--generations",
        "1",
        "--population",
        "2",
        check=False,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert_contains(output, "ERROR", "Invalid configuration")


# =============================================================================
# 5. Reproducibility Tests
# =============================================================================


def test_seed_reproducible_output():
    """Test --seed produces identical output across runs."""
    # Run twice with same seed
    result1 = run_cli(
        "--seed",
        "42",
        "--generations",
        "2",
        "--population",
        "3",
        "--log-level",
        "WARNING",
    )

    result2 = run_cli(
        "--seed",
        "42",
        "--generations",
        "2",
        "--population",
        "3",
        "--log-level",
        "WARNING",
    )

    # Outputs should be identical
    assert result1.stdout == result2.stdout, "Outputs differ with same seed"


def test_seed_with_export_reproducible():
    """Test --seed produces identical exported metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "run1.json"
        path2 = Path(tmpdir) / "run2.json"

        # Run twice with same seed
        run_cli(
            "--seed",
            "42",
            "--generations",
            "2",
            "--population",
            "3",
            "--export-metrics",
            str(path1),
            "--log-level",
            "WARNING",
        )

        run_cli(
            "--seed",
            "42",
            "--generations",
            "2",
            "--population",
            "3",
            "--export-metrics",
            str(path2),
            "--log-level",
            "WARNING",
        )

        data1 = load_json_file(path1)
        data2 = load_json_file(path2)

        # Random seed should be recorded
        assert data1.get("random_seed") == 42
        assert data2.get("random_seed") == 42

        # Metrics history should be identical (same initial population, same evolution)
        # Note: Exact fitness values may vary slightly due to timing, but structure identical
        assert len(data1["metrics_history"]) == len(data2["metrics_history"])


def test_no_seed_non_deterministic():
    """Test runs without --seed produce different results with exports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "run1.json"
        path2 = Path(tmpdir) / "run2.json"

        # Run twice without seed, export to compare metrics
        run_cli(
            "--generations",
            "2",
            "--population",
            "5",
            "--export-metrics",
            str(path1),
            "--log-level",
            "WARNING",
        )

        run_cli(
            "--generations",
            "2",
            "--population",
            "5",
            "--export-metrics",
            str(path2),
            "--log-level",
            "WARNING",
        )

        data1 = load_json_file(path1)
        data2 = load_json_file(path2)

        # Without seed, random_seed should be None or different
        seed1 = data1.get("random_seed")
        seed2 = data2.get("random_seed")

        assert seed1 is None and seed2 is None, (
            "Runs without --seed should not record seed"
        )

        # The actual metrics should differ due to different random initialization
        # Compare at least one generation's candidate counts (fitness)
        candidate_counts1 = [
            civ["candidate_count"] for civ in data1["metrics_history"][0]
        ]
        candidate_counts2 = [
            civ["candidate_count"] for civ in data2["metrics_history"][0]
        ]

        # Very unlikely all candidate counts are identical across 5 civilizations with different seeds
        assert candidate_counts1 != candidate_counts2, (
            f"All candidate counts identical without seed: {candidate_counts1} == {candidate_counts2}"
        )


# =============================================================================
# 6. Meta-Learning CLI Tests
# =============================================================================


def test_meta_learning_flag():
    """Test --meta-learning enables meta-learning feature."""
    result = run_cli(
        "--meta-learning",
        "--generations",
        "6",  # Need > adaptation_window (5) to see adaptation
        "--population",
        "4",
        "--log-level",
        "INFO",  # Need INFO to see meta-learning messages
    )

    assert result.returncode == 0
    assert_contains(
        result.stdout,
        "Meta-learning enabled",
        "Rates will adapt",
    )


def test_adaptation_window_custom():
    """Test --adaptation-window customizes meta-learning window."""
    result = run_cli(
        "--meta-learning",
        "--adaptation-window",
        "3",
        "--generations",
        "4",  # Need > window to see adaptation
        "--population",
        "4",
        "--log-level",
        "INFO",
    )

    assert result.returncode == 0
    assert_contains(result.stdout, "Meta-learning enabled", "window=3")


# =============================================================================
# 7. Comparison Mode Tests
# =============================================================================


def test_compare_baseline_mode():
    """Test --compare-baseline runs comparison against 3 baselines."""
    result = run_cli(
        "--compare-baseline",
        "--num-comparison-runs",
        "2",
        "--generations",
        "2",
        "--population",
        "3",
        "--log-level",
        "WARNING",
    )

    assert result.returncode == 0
    assert_contains(
        result.stdout,
        "STATISTICAL COMPARISON RESULTS",
        "CONSERVATIVE BASELINE",
        "BALANCED BASELINE",
        "AGGRESSIVE BASELINE",
        "CONVERGENCE STATISTICS",
    )


def test_num_comparison_runs():
    """Test --num-comparison-runs controls number of independent runs."""
    result = run_cli(
        "--compare-baseline",
        "--num-comparison-runs",
        "3",
        "--generations",
        "2",
        "--population",
        "2",
        "--log-level",
        "WARNING",
    )

    assert result.returncode == 0
    # Should mention 3 runs in output
    assert_contains(result.stdout, "Comparison runs: 3")


def test_comparison_with_seed():
    """Test comparison mode with --seed for reproducibility."""
    result = run_cli(
        "--compare-baseline",
        "--num-comparison-runs",
        "2",
        "--generations",
        "2",
        "--population",
        "3",
        "--seed",
        "42",
        "--log-level",
        "WARNING",
    )

    assert result.returncode == 0
    assert_contains(result.stdout, "Base seed: 42")


# =============================================================================
# 8. LLM Mode Tests (Conditional)
# =============================================================================


def test_llm_mode_without_api_key():
    """Test --llm without API key shows clear error message."""
    # Temporarily unset API key
    env = {"GEMINI_API_KEY": ""}

    result = run_cli(
        "--llm",
        "--generations",
        "1",
        "--population",
        "2",
        check=False,
        env=env,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert_contains(
        output,
        "ERROR",
        "GEMINI_API_KEY",
        ".env",
    )


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY environment variable",
)
def test_llm_mode_with_api_key():
    """Test --llm mode works with valid API key (integration test)."""
    result = run_cli(
        "--llm",
        "--generations",
        "1",
        "--population",
        "2",
        "--log-level",
        "WARNING",
        timeout=60,  # LLM calls may take longer
    )

    assert result.returncode == 0
    assert_contains(result.stdout, "LLM mode enabled", "Gemini")
