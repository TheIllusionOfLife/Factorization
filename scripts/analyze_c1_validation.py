#!/usr/bin/env python3
"""Statistical analysis script for C1 validation experiments.

Analyzes results from 30 experimental runs (10 collaborative, 10 search_only, 10 rulebased)
to test hypothesis H1a: Cognitive specialization with feedback produces emergence.

Success criteria:
- Emergence factor > 1.1 (collaborative outperforms baselines by >11%)
- Statistical significance: p < 0.05
- Effect size: Cohen's d ≥ 0.5 (medium effect)
"""

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def load_experiment_results(
    results_dir: Path, mode: str, seed_start: int, num_runs: int
) -> List[float]:
    """Load fitness values from experiment JSON files.

    Args:
        results_dir: Directory containing result files
        mode: Experiment mode (collaborative, search_only, rulebased)
        seed_start: Starting seed value for this mode
        num_runs: Number of runs to load

    Returns:
        List of final fitness values (best fitness from last generation)
    """
    fitness_values = []

    for i in range(num_runs):
        seed = seed_start + i
        filepath = results_dir / f"{mode}_seed{seed}.json"

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Extract final generation fitness (best of last generation)
            metrics_history = data["metrics_history"]
            if metrics_history:
                final_gen_metrics = metrics_history[-1]
                final_fitness = max(m["candidate_count"] for m in final_gen_metrics)
                fitness_values.append(final_fitness)
            else:
                print(f"⚠️  Warning: No metrics history in {filepath.name}")

        except FileNotFoundError:
            print(f"❌ Error: Missing file {filepath.name}")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"❌ Error parsing {filepath.name}: {e}")

    return fitness_values


def calculate_emergence_metrics(
    collaborative: List[float], search_only: List[float], rulebased: List[float]
) -> Dict[str, float]:
    """Calculate emergence metrics from fitness distributions.

    Args:
        collaborative: Fitness values from collaborative mode
        search_only: Fitness values from search_only baseline
        rulebased: Fitness values from rulebased baseline

    Returns:
        Dictionary with emergence metrics
    """
    collab_mean = statistics.mean(collaborative)
    search_only_mean = statistics.mean(search_only)
    rulebased_mean = statistics.mean(rulebased)

    # Emergence factor: collaborative / max(baselines)
    max_baseline = max(search_only_mean, rulebased_mean)
    emergence_factor = collab_mean / max_baseline if max_baseline > 0 else float("inf")

    # Synergy score: collaborative - max(baselines)
    synergy_score = collab_mean - max_baseline

    # Relative improvement percentage
    improvement_pct = (
        ((emergence_factor - 1.0) * 100) if max_baseline > 0 else float("inf")
    )

    return {
        "collaborative_mean": collab_mean,
        "search_only_mean": search_only_mean,
        "rulebased_mean": rulebased_mean,
        "max_baseline_mean": max_baseline,
        "emergence_factor": emergence_factor,
        "synergy_score": synergy_score,
        "improvement_pct": improvement_pct,
    }


def welch_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """Perform Welch's t-test (unequal variances t-test).

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Tuple of (t_statistic, p_value)
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1, var2 = statistics.variance(group1), statistics.variance(group2)

    # Welch's t-statistic
    t_stat = (mean1 - mean2) / math.sqrt(var1 / n1 + var2 / n2)

    # Welch-Satterthwaite degrees of freedom
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )

    # Approximate p-value using t-distribution CDF (simplified)
    # For small samples, this is an approximation
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df))

    return t_stat, p_value


def _t_cdf(t: float, df: float) -> float:
    """Approximate CDF of Student's t-distribution.

    Args:
        t: t-statistic value
        df: Degrees of freedom

    Returns:
        Cumulative probability P(T <= t)
    """
    # Simple approximation for t-distribution CDF
    # More accurate for df > 10
    x = df / (df + t**2)
    return 1 - 0.5 * _beta_inc(df / 2, 0.5, x)


def _beta_inc(a: float, b: float, x: float) -> float:
    """Approximate incomplete beta function (simplified).

    Args:
        a, b: Shape parameters
        x: Integration limit

    Returns:
        I_x(a, b) approximation
    """
    # Very simple approximation - good enough for our purposes
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use series approximation (first few terms)
    result = 0.0
    for k in range(20):
        coef = 1.0
        for j in range(k):
            coef *= (a + j) / (a + b + j)
        result += coef * (x ** (a + k)) * ((1 - x) ** b) / (a + k)

    return result / (x**a * (1 - x) ** b) if x > 0 and x < 1 else 0.0


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = statistics.variance(group1), statistics.variance(group2)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    return d


def calculate_confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for mean.

    Args:
        data: Sample data
        confidence: Confidence level (default 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(data)
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    sem = std / math.sqrt(n)  # Standard error of mean

    # t-value for 95% CI with df = n-1
    # Approximation: for n=10, df=9, t ≈ 2.262
    df = n - 1
    t_critical = 2.262 if df == 9 else 2.0  # Simplified for n=10

    interval = sem * t_critical

    return (mean - interval, mean + interval)


def test_h1a_criteria(
    collaborative: List[float], search_only: List[float], rulebased: List[float]
) -> Dict[str, any]:
    """Test H1a success criteria.

    Criteria:
    1. Emergence factor > 1.1 (11% improvement)
    2. Statistical significance: p < 0.05
    3. Effect size: Cohen's d ≥ 0.5

    Args:
        collaborative: Collaborative mode fitness values
        search_only: Search-only baseline fitness values
        rulebased: Rulebased baseline fitness values

    Returns:
        Dictionary with test results and pass/fail status
    """
    # Calculate emergence metrics
    metrics = calculate_emergence_metrics(collaborative, search_only, rulebased)

    # Determine which baseline is stronger (for hypothesis testing)
    if metrics["search_only_mean"] >= metrics["rulebased_mean"]:
        stronger_baseline = search_only
        stronger_baseline_name = "search_only"
    else:
        stronger_baseline = rulebased
        stronger_baseline_name = "rulebased"

    # Statistical tests vs stronger baseline
    t_stat, p_value = welch_t_test(collaborative, stronger_baseline)
    effect_size = cohens_d(collaborative, stronger_baseline)

    # Confidence intervals
    collab_ci = calculate_confidence_interval(collaborative)
    baseline_ci = calculate_confidence_interval(stronger_baseline)

    # Test criteria
    criterion_1 = metrics["emergence_factor"] > 1.1
    criterion_2 = p_value < 0.05
    criterion_3 = effect_size >= 0.5

    h1a_success = criterion_1 and criterion_2 and criterion_3

    return {
        "metrics": metrics,
        "stronger_baseline": stronger_baseline_name,
        "t_statistic": t_stat,
        "p_value": p_value,
        "effect_size_d": effect_size,
        "collaborative_ci": collab_ci,
        "baseline_ci": baseline_ci,
        "criterion_1_emergence": criterion_1,
        "criterion_2_significance": criterion_2,
        "criterion_3_effect_size": criterion_3,
        "h1a_success": h1a_success,
    }


def print_results_table(
    results: Dict[str, any],
    collaborative: List[float],
    search_only: List[float],
    rulebased: List[float],
) -> None:
    """Print formatted results table.

    Args:
        results: H1a test results from test_h1a_criteria()
        collaborative: Collaborative fitness values (for summary stats)
        search_only: Search-only fitness values (for summary stats)
        rulebased: Rulebased fitness values (for summary stats)
    """
    print("\n" + "=" * 80)
    print("C1 VALIDATION RESULTS - H1a Hypothesis Testing")
    print("=" * 80)

    print("\n📊 DESCRIPTIVE STATISTICS")
    print("-" * 80)

    modes = [
        ("Collaborative", collaborative),
        ("Search-Only", search_only),
        ("Rulebased", rulebased),
    ]

    print(f"{'Mode':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    for name, data in modes:
        print(
            f"{name:<20} {statistics.mean(data):<12.2f} {statistics.stdev(data):<12.2f} "
            f"{min(data):<12.2f} {max(data):<12.2f}"
        )

    print("\n📈 EMERGENCE METRICS")
    print("-" * 80)
    m = results["metrics"]
    print(f"Emergence Factor:     {m['emergence_factor']:.4f}")
    print(f"Synergy Score:        {m['synergy_score']:.2f}")
    print(f"Improvement:          {m['improvement_pct']:.2f}%")
    print(f"Max Baseline:         {results['stronger_baseline']}")

    print("\n🔬 STATISTICAL TESTS (vs stronger baseline)")
    print("-" * 80)
    print(f"Welch's t-statistic:  {results['t_statistic']:.4f}")
    print(f"p-value:              {results['p_value']:.6f}")
    print(f"Cohen's d:            {results['effect_size_d']:.4f}")

    print("\n📊 CONFIDENCE INTERVALS (95%)")
    print("-" * 80)
    ci_collab = results["collaborative_ci"]
    ci_base = results["baseline_ci"]
    print(f"Collaborative:        [{ci_collab[0]:.2f}, {ci_collab[1]:.2f}]")
    print(
        f"{results['stronger_baseline'].capitalize():<21} [{ci_base[0]:.2f}, {ci_base[1]:.2f}]"
    )

    print("\n✅ H1a SUCCESS CRITERIA")
    print("-" * 80)

    criteria = [
        (
            "Emergence Factor > 1.1",
            results["criterion_1_emergence"],
            f"{m['emergence_factor']:.4f} {'>' if results['criterion_1_emergence'] else '≤'} 1.1",
        ),
        (
            "Statistical Significance (p < 0.05)",
            results["criterion_2_significance"],
            f"p = {results['p_value']:.6f}",
        ),
        (
            "Effect Size (d ≥ 0.5)",
            results["criterion_3_effect_size"],
            f"d = {results['effect_size_d']:.4f}",
        ),
    ]

    for name, passed, detail in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<40} {status:<10} ({detail})")

    print("\n" + "=" * 80)
    if results["h1a_success"]:
        print("🎉 H1a HYPOTHESIS: SUPPORTED")
        print(
            "   Cognitive specialization with feedback produces significant emergence!"
        )
    else:
        print("❌ H1a HYPOTHESIS: NOT SUPPORTED")
        print("   Collaborative mode did not meet all success criteria.")
    print("=" * 80 + "\n")


def main():
    """Main analysis workflow."""
    # Configuration
    results_dir = Path("results/c1_validation")

    print("🔬 Loading C1 validation experiment results...")
    print(f"   Results directory: {results_dir}")

    # Load data
    collaborative = load_experiment_results(results_dir, "collaborative", 6000, 10)
    search_only = load_experiment_results(results_dir, "search_only", 7000, 10)
    rulebased = load_experiment_results(results_dir, "rulebased", 8000, 10)

    print(
        f"\n   Loaded: {len(collaborative)} collaborative, {len(search_only)} search_only, "
        f"{len(rulebased)} rulebased runs"
    )

    if len(collaborative) < 10 or len(search_only) < 10 or len(rulebased) < 10:
        print("\n⚠️  Warning: Some experiment files missing. Results may be incomplete.")

    # Analyze
    print("\n🧮 Computing statistical tests...")
    results = test_h1a_criteria(collaborative, search_only, rulebased)

    # Report
    print_results_table(results, collaborative, search_only, rulebased)

    # Export results
    output_file = results_dir / "h1a_analysis.json"
    with open(output_file, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        export_data = {
            "collaborative_fitness": [float(x) for x in collaborative],
            "search_only_fitness": [float(x) for x in search_only],
            "rulebased_fitness": [float(x) for x in rulebased],
            "metrics": {k: float(v) for k, v in results["metrics"].items()},
            "statistical_tests": {
                "stronger_baseline": results["stronger_baseline"],
                "t_statistic": float(results["t_statistic"]),
                "p_value": float(results["p_value"]),
                "effect_size_d": float(results["effect_size_d"]),
            },
            "confidence_intervals": {
                "collaborative": [float(x) for x in results["collaborative_ci"]],
                "baseline": [float(x) for x in results["baseline_ci"]],
            },
            "h1a_criteria": {
                "emergence_factor": results["criterion_1_emergence"],
                "significance": results["criterion_2_significance"],
                "effect_size": results["criterion_3_effect_size"],
                "overall_success": results["h1a_success"],
            },
        }
        json.dump(export_data, f, indent=2)

    print(f"💾 Results exported to: {output_file}\n")


if __name__ == "__main__":
    main()
