#!/usr/bin/env python3
"""Statistical analysis script for C2 validation experiments.

Analyzes results from C2 LLM-guided mutations vs C1 rule-based feedback and rulebased baseline
to test hypothesis H1b: LLM reasoning enhances collaborative evolution beyond rule-based feedback.

Success criteria:
- Emergence factor > 1.1 (C2 outperforms baselines by >11%)
- Statistical significance: p < 0.05 (vs C1 and rulebased)
- Effect size: Cohen's d ≥ 0.5 (medium effect)
- Improvement over C1: Shows value of LLM reasoning

Comparison structure:
- C2 (LLM-guided) vs C1 (rule-based collaborative)
- C2 (LLM-guided) vs Rulebased baseline
- C1 vs Rulebased (for reference)
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
        mode: Experiment mode filename pattern (e.g., "c2_llm", "collaborative", "rulebased")
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

            # Extract fitness - support multiple formats
            if "metrics_history" in data and data["metrics_history"]:
                # Old format: extract best fitness from last generation
                final_gen_metrics = data["metrics_history"][-1]
                final_fitness = max(m["candidate_count"] for m in final_gen_metrics)
                fitness_values.append(final_fitness)
            elif "best_fitness" in data:
                # New format: best_fitness field
                final_fitness = float(data["best_fitness"])
                fitness_values.append(final_fitness)
            elif "final_fitness" in data:
                # Alternative format: final_fitness field
                final_fitness = float(data["final_fitness"])
                fitness_values.append(final_fitness)
            else:
                print(
                    f"⚠️  Warning: No fitness data in {filepath.name} (missing metrics_history, best_fitness, and final_fitness)"
                )

        except FileNotFoundError:
            print(f"❌ Error: Missing file {filepath.name}")
        except (KeyError, json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"❌ Error parsing {filepath.name}: {e}")

    return fitness_values


def calculate_emergence_metrics(
    c2_llm: List[float], c1_rulebased: List[float], rulebased: List[float]
) -> Dict[str, float]:
    """Calculate emergence metrics for C2 validation.

    Args:
        c2_llm: Fitness values from C2 LLM-guided mode
        c1_rulebased: Fitness values from C1 rule-based collaborative
        rulebased: Fitness values from rulebased baseline

    Returns:
        Dictionary with emergence metrics
    """
    c2_mean = statistics.mean(c2_llm)
    c1_mean = statistics.mean(c1_rulebased)
    rulebased_mean = statistics.mean(rulebased)

    # Emergence factor: C2 / max(C1, rulebased)
    max_baseline = max(c1_mean, rulebased_mean)
    emergence_factor = c2_mean / max_baseline if max_baseline > 0 else float("inf")

    # Synergy score: C2 - max(baselines)
    synergy_score = c2_mean - max_baseline

    # Relative improvement percentage
    improvement_pct = (
        ((emergence_factor - 1.0) * 100) if max_baseline > 0 else float("inf")
    )

    # C2 vs C1 improvement (key metric for H1b)
    c2_vs_c1_improvement = (
        ((c2_mean / c1_mean) - 1.0) * 100 if c1_mean > 0 else float("inf")
    )

    # C1 emergence factor (for reference)
    c1_emergence = c1_mean / rulebased_mean if rulebased_mean > 0 else float("inf")

    return {
        "c2_llm_mean": c2_mean,
        "c1_rulebased_mean": c1_mean,
        "rulebased_mean": rulebased_mean,
        "max_baseline_mean": max_baseline,
        "emergence_factor_c2": emergence_factor,
        "synergy_score": synergy_score,
        "improvement_pct": improvement_pct,
        "c2_vs_c1_improvement_pct": c2_vs_c1_improvement,
        "c1_emergence_factor": c1_emergence,
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

    # Approximate p-value using t-distribution CDF
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df))

    return t_stat, p_value


def _t_cdf(t: float, df: float) -> float:
    """Approximate CDF of Student's t-distribution."""
    x = df / (df + t**2)
    return 1 - 0.5 * _beta_inc(df / 2, 0.5, x)


def _beta_inc(a: float, b: float, x: float) -> float:
    """Approximate incomplete beta function."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use series approximation
    result = 0.0
    for k in range(20):
        coef = 1.0
        for j in range(k):
            coef *= (a + j) / (a + b + j)
        result += coef * (x ** (a + k)) * ((1 - x) ** b) / (a + k)

    return result / (x**a * (1 - x) ** b) if 0 < x < 1 else 0.0


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

    # t-value for 95% CI (approximate)
    df = n - 1
    t_critical = 2.131 if df == 14 else 2.262 if df == 9 else 2.0

    interval = sem * t_critical

    return (mean - interval, mean + interval)


def test_h1b_criteria(
    c2_llm: List[float], c1_rulebased: List[float], rulebased: List[float]
) -> Dict[str, any]:
    """Test H1b success criteria.

    H1b Hypothesis: LLM-guided mutations produce emergence where rule-based failed.

    Criteria:
    1. Emergence factor > 1.1 (11% improvement over best baseline)
    2. Statistical significance: p < 0.05 (C2 vs C1 AND C2 vs rulebased)
    3. Effect size: Cohen's d ≥ 0.5 (medium effect vs C1)
    4. Improvement over C1: C2 > C1 (LLM adds value)

    Args:
        c2_llm: C2 LLM-guided mode fitness values
        c1_rulebased: C1 rule-based collaborative fitness values
        rulebased: Rulebased baseline fitness values

    Returns:
        Dictionary with test results and pass/fail status
    """
    # Calculate emergence metrics
    metrics = calculate_emergence_metrics(c2_llm, c1_rulebased, rulebased)

    # Determine which baseline is stronger
    if metrics["c1_rulebased_mean"] >= metrics["rulebased_mean"]:
        stronger_baseline = c1_rulebased
        stronger_baseline_name = "c1_rulebased"
    else:
        stronger_baseline = rulebased
        stronger_baseline_name = "rulebased"

    # Statistical tests
    t_vs_stronger, p_vs_stronger = welch_t_test(c2_llm, stronger_baseline)
    d_vs_stronger = cohens_d(c2_llm, stronger_baseline)

    # C2 vs C1 comparison (key for H1b)
    t_c2_vs_c1, p_c2_vs_c1 = welch_t_test(c2_llm, c1_rulebased)
    d_c2_vs_c1 = cohens_d(c2_llm, c1_rulebased)

    # C2 vs rulebased comparison
    t_c2_vs_rb, p_c2_vs_rb = welch_t_test(c2_llm, rulebased)
    d_c2_vs_rb = cohens_d(c2_llm, rulebased)

    # Confidence intervals
    c2_ci = calculate_confidence_interval(c2_llm)
    c1_ci = calculate_confidence_interval(c1_rulebased)
    rb_ci = calculate_confidence_interval(rulebased)

    # Test H1b criteria
    criterion_1_emergence = metrics["emergence_factor_c2"] > 1.1
    criterion_2_significance = p_vs_stronger < 0.05
    criterion_3_effect_size = d_vs_stronger >= 0.5
    criterion_4_improvement_c1 = metrics["c2_llm_mean"] > metrics["c1_rulebased_mean"]

    h1b_success = (
        criterion_1_emergence
        and criterion_2_significance
        and criterion_3_effect_size
        and criterion_4_improvement_c1
    )

    return {
        "metrics": metrics,
        "stronger_baseline": stronger_baseline_name,
        "t_vs_stronger": t_vs_stronger,
        "p_vs_stronger": p_vs_stronger,
        "d_vs_stronger": d_vs_stronger,
        "t_c2_vs_c1": t_c2_vs_c1,
        "p_c2_vs_c1": p_c2_vs_c1,
        "d_c2_vs_c1": d_c2_vs_c1,
        "t_c2_vs_rb": t_c2_vs_rb,
        "p_c2_vs_rb": p_c2_vs_rb,
        "d_c2_vs_rb": d_c2_vs_rb,
        "c2_ci": c2_ci,
        "c1_ci": c1_ci,
        "rulebased_ci": rb_ci,
        "criterion_1_emergence": criterion_1_emergence,
        "criterion_2_significance": criterion_2_significance,
        "criterion_3_effect_size": criterion_3_effect_size,
        "criterion_4_improvement_c1": criterion_4_improvement_c1,
        "h1b_success": h1b_success,
    }


def print_results_table(
    results: Dict[str, any],
    c2_llm: List[float],
    c1_rulebased: List[float],
    rulebased: List[float],
) -> None:
    """Print formatted results table for C2 validation.

    Args:
        results: H1b test results from test_h1b_criteria()
        c2_llm: C2 LLM-guided fitness values
        c1_rulebased: C1 rule-based fitness values
        rulebased: Rulebased baseline fitness values
    """
    print("\n" + "=" * 80)
    print("C2 VALIDATION RESULTS - H1b Hypothesis Testing")
    print("=" * 80)

    print("\n📊 DESCRIPTIVE STATISTICS")
    print("-" * 80)

    modes = [
        ("C2 LLM-Guided", c2_llm),
        ("C1 Rule-Based", c1_rulebased),
        ("Rulebased Baseline", rulebased),
    ]

    print(f"{'Mode':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    for name, data in modes:
        print(
            f"{name:<25} {statistics.mean(data):<12.2f} {statistics.stdev(data):<12.2f} "
            f"{min(data):<12.2f} {max(data):<12.2f}"
        )

    print("\n📈 EMERGENCE METRICS")
    print("-" * 80)
    m = results["metrics"]
    print(f"C2 Emergence Factor:      {m['emergence_factor_c2']:.4f}")
    print(f"C1 Emergence Factor:      {m['c1_emergence_factor']:.4f} (reference)")
    print(f"C2 vs C1 Improvement:     {m['c2_vs_c1_improvement_pct']:+.2f}%")
    print(f"Synergy Score:            {m['synergy_score']:.2f}")
    print(f"Overall Improvement:      {m['improvement_pct']:+.2f}%")
    print(f"Stronger Baseline:        {results['stronger_baseline']}")

    print("\n🔬 STATISTICAL TESTS")
    print("-" * 80)

    # C2 vs C1 (key comparison for H1b)
    print("C2 vs C1 (Rule-Based):")
    print(f"  t-statistic:            {results['t_c2_vs_c1']:.4f}")
    print(f"  p-value:                {results['p_c2_vs_c1']:.6f}")
    print(f"  Cohen's d:              {results['d_c2_vs_c1']:.4f}")

    # C2 vs Rulebased
    print("\nC2 vs Rulebased Baseline:")
    print(f"  t-statistic:            {results['t_c2_vs_rb']:.4f}")
    print(f"  p-value:                {results['p_c2_vs_rb']:.6f}")
    print(f"  Cohen's d:              {results['d_c2_vs_rb']:.4f}")

    # C2 vs Stronger baseline
    print(f"\nC2 vs Stronger Baseline ({results['stronger_baseline']}):")
    print(f"  t-statistic:            {results['t_vs_stronger']:.4f}")
    print(f"  p-value:                {results['p_vs_stronger']:.6f}")
    print(f"  Cohen's d:              {results['d_vs_stronger']:.4f}")

    print("\n📊 CONFIDENCE INTERVALS (95%)")
    print("-" * 80)
    print(
        f"C2 LLM-Guided:            [{results['c2_ci'][0]:.2f}, {results['c2_ci'][1]:.2f}]"
    )
    print(
        f"C1 Rule-Based:            [{results['c1_ci'][0]:.2f}, {results['c1_ci'][1]:.2f}]"
    )
    print(
        f"Rulebased Baseline:       [{results['rulebased_ci'][0]:.2f}, {results['rulebased_ci'][1]:.2f}]"
    )

    print("\n✅ H1b SUCCESS CRITERIA")
    print("-" * 80)

    criteria = [
        (
            "Emergence Factor > 1.1",
            results["criterion_1_emergence"],
            f"{m['emergence_factor_c2']:.4f} {'>' if results['criterion_1_emergence'] else '≤'} 1.1",
        ),
        (
            "Statistical Significance (p < 0.05)",
            results["criterion_2_significance"],
            f"p = {results['p_vs_stronger']:.6f} vs {results['stronger_baseline']}",
        ),
        (
            "Effect Size (d ≥ 0.5)",
            results["criterion_3_effect_size"],
            f"d = {results['d_vs_stronger']:.4f}",
        ),
        (
            "Improvement over C1",
            results["criterion_4_improvement_c1"],
            f"C2 {'+' if results['criterion_4_improvement_c1'] else ''}{m['c2_vs_c1_improvement_pct']:.2f}% vs C1",
        ),
    ]

    for name, passed, detail in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<40} {status:<10} ({detail})")

    print("\n" + "=" * 80)
    if results["h1b_success"]:
        print("🎉 H1b HYPOTHESIS: SUPPORTED")
        print(
            "   LLM reasoning enhances collaborative evolution beyond rule-based feedback!"
        )
    else:
        print("❌ H1b HYPOTHESIS: NOT SUPPORTED")
        print("   C2 LLM-guided mode did not meet all success criteria.")
    print("=" * 80 + "\n")


def main():
    """Main analysis workflow for C2 validation."""
    print("🔬 C2 Validation Analysis - LLM-Guided vs Rule-Based Collaborative")
    print("=" * 80)

    # Configuration
    c2_dir = Path("results/c2_validation")
    c1_dir = Path("results/c1_validation")

    print("\n📁 Loading experiment results...")
    print(f"   C2 results: {c2_dir}")
    print(f"   C1 results: {c1_dir}")

    # Load C2 data (15 runs, seeds 9000-9014)
    c2_llm = load_experiment_results(c2_dir, "c2_llm", 9000, 15)

    # Load C1 data (10 runs, seeds 6000-6009) and rulebased (10 runs, seeds 8000-8009)
    c1_rulebased = load_experiment_results(c1_dir, "collaborative", 6000, 10)
    rulebased = load_experiment_results(c1_dir, "rulebased", 8000, 10)

    print(
        f"\n   Loaded: {len(c2_llm)} C2 LLM, {len(c1_rulebased)} C1 rule-based, "
        f"{len(rulebased)} rulebased runs"
    )

    if len(c2_llm) < 15:
        print(
            "\n⚠️  Warning: Some C2 experiment files missing. Results may be incomplete."
        )
    if len(c1_rulebased) < 10 or len(rulebased) < 10:
        print(
            "\n⚠️  Warning: Some C1/baseline files missing. Results may be incomplete."
        )

    # Analyze
    print("\n🧮 Computing statistical tests...")
    results = test_h1b_criteria(c2_llm, c1_rulebased, rulebased)

    # Report
    print_results_table(results, c2_llm, c1_rulebased, rulebased)

    # Export results
    output_file = c2_dir / "h1b_analysis.json"
    with open(output_file, "w") as f:
        export_data = {
            "c2_llm_fitness": [float(x) for x in c2_llm],
            "c1_rulebased_fitness": [float(x) for x in c1_rulebased],
            "rulebased_fitness": [float(x) for x in rulebased],
            "metrics": {k: float(v) for k, v in results["metrics"].items()},
            "statistical_tests": {
                "stronger_baseline": results["stronger_baseline"],
                "c2_vs_c1": {
                    "t_statistic": float(results["t_c2_vs_c1"]),
                    "p_value": float(results["p_c2_vs_c1"]),
                    "effect_size_d": float(results["d_c2_vs_c1"]),
                },
                "c2_vs_rulebased": {
                    "t_statistic": float(results["t_c2_vs_rb"]),
                    "p_value": float(results["p_c2_vs_rb"]),
                    "effect_size_d": float(results["d_c2_vs_rb"]),
                },
                "c2_vs_stronger": {
                    "t_statistic": float(results["t_vs_stronger"]),
                    "p_value": float(results["p_vs_stronger"]),
                    "effect_size_d": float(results["d_vs_stronger"]),
                },
            },
            "confidence_intervals": {
                "c2_llm": [float(x) for x in results["c2_ci"]],
                "c1_rulebased": [float(x) for x in results["c1_ci"]],
                "rulebased": [float(x) for x in results["rulebased_ci"]],
            },
            "h1b_criteria": {
                "emergence_factor": results["criterion_1_emergence"],
                "significance": results["criterion_2_significance"],
                "effect_size": results["criterion_3_effect_size"],
                "improvement_over_c1": results["criterion_4_improvement_c1"],
                "overall_success": results["h1b_success"],
            },
        }
        json.dump(export_data, f, indent=2)

    print(f"💾 Results exported to: {output_file}\n")


if __name__ == "__main__":
    main()
