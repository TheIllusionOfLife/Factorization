#!/usr/bin/env python3
"""
Manual statistical analysis of Prometheus benchmarks.

This script analyzes multiple Prometheus benchmark runs to determine
if collaborative mode shows statistically significant emergence compared
to baseline modes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.statistics import StatisticalAnalyzer

# Decision thresholds (extracted as named constants per CLAUDE.md)
EMERGENCE_THRESHOLD_STRONG = 1.20  # >20% improvement for strong success
EMERGENCE_THRESHOLD_MODERATE = 1.10  # 10-20% improvement for moderate success
EMERGENCE_THRESHOLD_WEAK = 1.05  # 5-10% improvement for weak signal
EMERGENCE_THRESHOLD_NEUTRAL = 1.00  # 0-5% improvement (no emergence)

P_VALUE_THRESHOLD_HIGH = 0.01  # Highly significant
P_VALUE_THRESHOLD_MODERATE = 0.05  # Significant
P_VALUE_THRESHOLD_WEAK = 0.10  # Marginally significant

COHENS_D_LARGE = 0.8  # Large effect size
COHENS_D_MEDIUM = 0.5  # Medium effect size
COHENS_D_SMALL = 0.3  # Small effect size


def load_benchmark_results(
    results_dir: Path, pattern: str = "prometheus_*.json"
) -> List[Dict]:
    """Load all benchmark JSON files matching pattern."""
    files = sorted(results_dir.glob(pattern))
    results = []

    for file in files:
        try:
            with file.open() as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {file}: {e}", file=sys.stderr)

    return results


def extract_fitness_by_mode(results: List[Dict]) -> Dict[str, List[float]]:
    """Extract best fitness values for each mode across all runs."""
    fitness_by_mode = {
        "collaborative": [],
        "search_only": [],
        "eval_only": [],
        "rulebased": [],
    }

    for result in results:
        mode_results = result.get("mode_results", {})
        for mode in fitness_by_mode:
            if mode in mode_results:
                fitness = mode_results[mode]["results"]["best_fitness"]
                fitness_by_mode[mode].append(fitness)

    return fitness_by_mode


def calculate_emergence_metrics(
    fitness_by_mode: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Calculate emergence factor and synergy score for each run."""
    n_runs = len(fitness_by_mode["collaborative"])

    emergence_factors = []
    synergy_scores = []

    for i in range(n_runs):
        collab = fitness_by_mode["collaborative"][i]
        search = fitness_by_mode["search_only"][i]
        eval_only = fitness_by_mode["eval_only"][i]
        rulebased = fitness_by_mode["rulebased"][i]

        baseline_max = max(search, eval_only, rulebased)

        emergence_factor = collab / baseline_max if baseline_max > 0 else 0
        synergy_score = collab - baseline_max

        emergence_factors.append(emergence_factor)
        synergy_scores.append(synergy_score)

    return {"emergence_factors": emergence_factors, "synergy_scores": synergy_scores}


def print_summary_statistics(
    fitness_by_mode: Dict[str, List[float]], emergence_metrics: Dict[str, List[float]]
):
    """Print descriptive statistics for all modes."""
    print("\n" + "=" * 70)
    print("📊 PROMETHEUS PHASE 1 BENCHMARK RESULTS")
    print("=" * 70 + "\n")

    print(f"Number of runs: {len(fitness_by_mode['collaborative'])}\n")

    # Fitness statistics per mode
    print("Best Fitness Statistics (across runs):\n")
    for mode in ["collaborative", "search_only", "eval_only", "rulebased"]:
        values = fitness_by_mode[mode]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            min_val = np.min(values)
            max_val = np.max(values)

            print(f"  {mode:15s}: mean={mean_val:>10,.1f}, std={std_val:>10,.1f}")
            print(
                f"                   median={median_val:>8,.1f}, min={min_val:>10,.1f}, max={max_val:>10,.1f}\n"
            )

    # Emergence metrics
    print("\nEmergence Metrics:\n")

    ef_values = emergence_metrics["emergence_factors"]
    if ef_values:
        print("  Emergence Factor (collaborative / max_baseline):")
        print(f"    mean={np.mean(ef_values):.4f}, std={np.std(ef_values):.4f}")
        print(
            f"    median={np.median(ef_values):.4f}, min={np.min(ef_values):.4f}, max={np.max(ef_values):.4f}"
        )

        if np.mean(ef_values) < EMERGENCE_THRESHOLD_NEUTRAL:
            print(
                "    ⚠️  WARNING: Emergence factor < 1.0 indicates collaborative UNDERPERFORMS"
            )
        elif np.mean(ef_values) < EMERGENCE_THRESHOLD_WEAK:
            print("    ℹ️  Emergence factor < 1.05 indicates minimal/no emergence")
        elif np.mean(ef_values) < EMERGENCE_THRESHOLD_MODERATE:
            print("    ✓ Weak emergence signal (5-10% improvement)")
        else:
            print("    ✓✓ Strong emergence signal (>10% improvement)")

    ss_values = emergence_metrics["synergy_scores"]
    if ss_values:
        print("\n  Synergy Score (collaborative - max_baseline):")
        print(f"    mean={np.mean(ss_values):>10,.1f}, std={np.std(ss_values):>10,.1f}")
        print(f"    median={np.median(ss_values):>8,.1f}")


def perform_statistical_tests(fitness_by_mode: Dict[str, List[float]]):
    """Perform statistical comparisons between collaborative and baselines."""
    print("\n" + "=" * 70)
    print("📈 STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70 + "\n")

    analyzer = StatisticalAnalyzer()

    collab = fitness_by_mode["collaborative"]

    # Test against each baseline
    for baseline_name in ["search_only", "eval_only", "rulebased"]:
        baseline = fitness_by_mode[baseline_name]

        if not collab or not baseline:
            print(f"⚠️  Skipping {baseline_name}: insufficient data\n")
            continue

        result = analyzer.compare_fitness_distributions(
            evolved_scores=collab, baseline_scores=baseline
        )

        print(f"Collaborative vs {baseline_name.upper()}:")
        print(f"  Collaborative mean: {np.mean(collab):>10,.1f}")
        print(f"  {baseline_name:15s} mean: {np.mean(baseline):>10,.1f}")

        improvement_pct = (
            ((np.mean(collab) / np.mean(baseline)) - 1) * 100
            if np.mean(baseline) > 0
            else float("inf")
        )
        print(f"  Improvement:        {improvement_pct:>9.1f}%")

        print(f"  p-value:            {result.p_value:.6f}", end="")
        if result.p_value < 0.001:
            print(" ***")
        elif result.p_value < 0.01:
            print(" **")
        elif result.p_value < 0.05:
            print(" *")
        else:
            print(" (not significant)")

        print(f"  Cohen's d:          {result.effect_size:>9.3f}")
        print(
            f"  95% CI:             [{result.confidence_interval[0]:>10,.1f}, {result.confidence_interval[1]:>10,.1f}]"
        )
        print(f"  Significant:        {result.is_significant}")
        print()


def make_recommendation(
    fitness_by_mode: Dict[str, List[float]], emergence_metrics: Dict[str, List[float]]
):
    """Apply decision framework and make recommendation."""
    print("\n" + "=" * 70)
    print("🎯 PHASE 2 DECISION RECOMMENDATION")
    print("=" * 70 + "\n")

    # Calculate key metrics
    analyzer = StatisticalAnalyzer()
    collab = fitness_by_mode["collaborative"]
    rulebased = fitness_by_mode["rulebased"]

    emergence_factor_mean = np.mean(emergence_metrics["emergence_factors"])

    if not collab or not rulebased:
        print("❌ ERROR: Insufficient data for recommendation\n")
        return

    result = analyzer.compare_fitness_distributions(
        evolved_scores=collab, baseline_scores=rulebased
    )

    p_value = result.p_value
    cohens_d = result.effect_size

    print("Key Metrics:")
    print(f"  Emergence Factor: {emergence_factor_mean:.4f}")
    print(f"  p-value (vs rulebased): {p_value:.6f}")
    print(f"  Cohen's d (vs rulebased): {cohens_d:.3f}\n")

    # Apply decision matrix
    print("Decision Matrix Application:\n")

    if (
        emergence_factor_mean >= EMERGENCE_THRESHOLD_STRONG
        and p_value < P_VALUE_THRESHOLD_HIGH
        and cohens_d >= COHENS_D_LARGE
    ):
        decision = "GO_PHASE_2_LLM"
        print("✅ **STRONG SUCCESS** → GO to Phase 2 LLM Integration")
        print("   - Collaborative shows strong emergence (>20% improvement)")
        print("   - Highly significant (p<0.01) with large effect size (d≥0.8)")
        print("   - Next Steps: Prompt engineering experiments → Phase 2 LLM")
        print("   - Estimated Cost: ~$0.10 for prompts, then ~$5 for Phase 2")
        print("   - Timeline: 2-3 hours prompt optimization, then 4 weeks Phase 2")

    elif (
        emergence_factor_mean >= EMERGENCE_THRESHOLD_MODERATE
        and p_value < P_VALUE_THRESHOLD_MODERATE
        and cohens_d >= COHENS_D_MEDIUM
    ):
        decision = "GO_WITH_CAUTION"
        print("⚠️  **MODERATE SUCCESS** → GO with Caution (prompt engineering first)")
        print("   - Collaborative shows moderate emergence (10-20% improvement)")
        print("   - Significant (p<0.05) with medium effect size (d≥0.5)")
        print("   - Next Steps: Extensive prompt engineering before full Phase 2")
        print("   - Estimated Cost: ~$0.20 for prompt experiments")
        print("   - Timeline: 1 week prompt optimization, re-evaluate")

    elif (
        emergence_factor_mean >= EMERGENCE_THRESHOLD_WEAK
        and p_value < P_VALUE_THRESHOLD_WEAK
        and cohens_d >= COHENS_D_SMALL
    ):
        decision = "CONDITIONAL_EXTEND"
        print("ℹ️  **WEAK SIGNAL** → Conditional (extend experiments)")
        print("   - Weak emergence signal (5-10% improvement)")
        print("   - Marginally significant with small effect")
        print("   - Next Steps: Extend to 10-15 runs for better statistical power")
        print("   - Estimated Cost: $0 (rule-based only)")
        print("   - Timeline: +1 week for extended benchmarks")

    elif emergence_factor_mean >= EMERGENCE_THRESHOLD_NEUTRAL:
        decision = "PIVOT_META_LEARNING"
        print("⛔ **NO EMERGENCE** → PIVOT to Meta-Learning")
        print("   - No meaningful emergence detected (0-5% improvement)")
        print("   - Not statistically significant or very small effect")
        print("   - Next Steps: Validate meta-learning system instead")
        print("   - Estimated Cost: $0 (rule-based only)")
        print("   - Timeline: 2 weeks meta-learning validation")

    else:
        decision = "ANALYZE_FAILURE"
        print("❌ **NEGATIVE RESULT** → Analyze Failure")
        print("   - Collaborative UNDERPERFORMS baselines")
        print("   - Emergence factor < 1.0 indicates systematic issue")
        print("   - Next Steps: Root cause analysis, possible redesign")
        print("   - Estimated Cost: $0")
        print("   - Timeline: 3-5 days investigation")
        print("\n   Possible Issues:")
        print("     • Agent communication overhead too high")
        print("     • Feedback loop not improving strategies")
        print("     • SearchSpecialist not utilizing EvaluationSpecialist feedback")
        print("     • Message passing introducing errors/noise")

    print(f"\n📋 DECISION: {decision}")
    print()


def main():
    """Main analysis workflow."""
    results_dir = Path("results/benchmarks")

    if not results_dir.exists():
        print(f"❌ Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load results
    results = load_benchmark_results(results_dir)

    if not results:
        print("❌ Error: No benchmark results found", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Loaded {len(results)} benchmark result(s)")

    # Extract fitness data
    fitness_by_mode = extract_fitness_by_mode(results)

    # Calculate emergence metrics
    emergence_metrics = calculate_emergence_metrics(fitness_by_mode)

    # Print summary statistics
    print_summary_statistics(fitness_by_mode, emergence_metrics)

    # Perform statistical tests
    perform_statistical_tests(fitness_by_mode)

    # Make recommendation
    make_recommendation(fitness_by_mode, emergence_metrics)


if __name__ == "__main__":
    main()
