#!/usr/bin/env python3
"""
Aggregate Results Script

Aggregates multiple experimental result files into a single analysis-ready format.
Computes descriptive statistics, performs statistical tests, and prepares data for visualization.

Usage:
    python scripts/aggregate_results.py --results-dir results/ --output results/aggregated_results.json

Author: Generated for Factorization project
Date: 2025-10-31
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.statistics import StatisticalAnalyzer, ConvergenceDetector


@dataclass
class AggregatedCondition:
    """Statistics for a single experimental condition."""
    condition_name: str
    n_runs: int
    n_valid: int  # Runs without errors

    # Final fitness statistics
    fitness_mean: float
    fitness_std: float
    fitness_median: float
    fitness_min: float
    fitness_max: float
    fitness_ci_lower: float
    fitness_ci_upper: float

    # Convergence statistics
    convergence_rate: float  # Proportion of runs that converged
    mean_generations_to_converge: Optional[float]
    std_generations_to_converge: Optional[float]

    # Raw data
    fitness_values: List[float]
    convergence_generations: List[Optional[int]]
    seeds: List[int]


@dataclass
class ComparisonResult:
    """Results of statistical comparison between two conditions."""
    condition_a: str
    condition_b: str
    mean_difference: float
    improvement_pct: float
    p_value: float
    effect_size: float  # Cohen's d
    ci_lower: float
    ci_upper: float
    significant: bool  # p < 0.05
    interpretation: str  # Small/Medium/Large effect


def load_comparison_results(results_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    """
    Load all JSON result files matching pattern.

    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern to match files (e.g., "rulebased_run_*.json")

    Returns:
        List of parsed JSON objects
    """
    result_files = sorted(results_dir.glob(pattern))

    if not result_files:
        print(f"⚠️  Warning: No files found matching pattern '{pattern}' in {results_dir}")
        return []

    print(f"📁 Loading {len(result_files)} files matching '{pattern}'...")

    results = []
    failed = []

    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except json.JSONDecodeError as e:
            failed.append((file_path.name, str(e)))
        except Exception as e:
            failed.append((file_path.name, str(e)))

    if failed:
        print(f"❌ Failed to load {len(failed)} files:")
        for name, error in failed:
            print(f"   - {name}: {error}")

    print(f"✅ Successfully loaded {len(results)} files\n")
    return results


def extract_fitness_values(results: List[Dict[str, Any]]) -> List[float]:
    """
    Extract final fitness values from comparison results.

    Args:
        results: List of comparison result dictionaries

    Returns:
        List of final fitness values (last generation, best strategy)
    """
    fitness_values = []

    for result in results:
        try:
            # Each result has multiple comparison runs, we want the final best fitness
            # Structure: result['runs'][run_index]['evolved_fitness'][-1]
            runs = result.get('runs', [])
            if runs:
                # Take the first run's final fitness (each file should have 1 run)
                evolved_fitness = runs[0].get('evolved_fitness', [])
                if evolved_fitness:
                    fitness_values.append(evolved_fitness[-1])
        except (KeyError, IndexError, TypeError) as e:
            print(f"⚠️  Warning: Could not extract fitness from result: {e}")

    return fitness_values


def extract_convergence_info(results: List[Dict[str, Any]]) -> List[Optional[int]]:
    """
    Extract convergence generation for each run.

    Args:
        results: List of comparison result dictionaries

    Returns:
        List of convergence generations (None if didn't converge)
    """
    convergence_gens = []

    for result in results:
        try:
            runs = result.get('runs', [])
            if runs:
                gen = runs[0].get('generations_to_convergence')
                convergence_gens.append(gen)
        except (KeyError, IndexError, TypeError):
            convergence_gens.append(None)

    return convergence_gens


def extract_seeds(results: List[Dict[str, Any]]) -> List[int]:
    """Extract random seeds from results."""
    seeds = []

    for result in results:
        try:
            runs = result.get('runs', [])
            if runs:
                seed = runs[0].get('random_seed')
                if seed is not None:
                    seeds.append(seed)
        except (KeyError, IndexError, TypeError):
            pass

    return seeds


def compute_aggregated_stats(
    condition_name: str,
    results: List[Dict[str, Any]]
) -> AggregatedCondition:
    """
    Compute aggregated statistics for a condition.

    Args:
        condition_name: Name of the condition (e.g., "Rule-Based", "LLM")
        results: List of result dictionaries

    Returns:
        AggregatedCondition with computed statistics
    """
    fitness_values = extract_fitness_values(results)
    convergence_gens = extract_convergence_info(results)
    seeds = extract_seeds(results)

    n_runs = len(results)
    n_valid = len(fitness_values)

    if n_valid == 0:
        print(f"❌ Error: No valid fitness values for condition '{condition_name}'")
        return None

    # Fitness statistics
    fitness_array = np.array(fitness_values)
    fitness_mean = float(np.mean(fitness_array))
    fitness_std = float(np.std(fitness_array, ddof=1))
    fitness_median = float(np.median(fitness_array))
    fitness_min = float(np.min(fitness_array))
    fitness_max = float(np.max(fitness_array))

    # 95% confidence interval
    ci_margin = 1.96 * (fitness_std / np.sqrt(n_valid))
    fitness_ci_lower = fitness_mean - ci_margin
    fitness_ci_upper = fitness_mean + ci_margin

    # Convergence statistics
    converged_runs = [g for g in convergence_gens if g is not None]
    convergence_rate = len(converged_runs) / n_valid if n_valid > 0 else 0.0

    if converged_runs:
        mean_gens = float(np.mean(converged_runs))
        std_gens = float(np.std(converged_runs, ddof=1)) if len(converged_runs) > 1 else 0.0
    else:
        mean_gens = None
        std_gens = None

    return AggregatedCondition(
        condition_name=condition_name,
        n_runs=n_runs,
        n_valid=n_valid,
        fitness_mean=fitness_mean,
        fitness_std=fitness_std,
        fitness_median=fitness_median,
        fitness_min=fitness_min,
        fitness_max=fitness_max,
        fitness_ci_lower=fitness_ci_lower,
        fitness_ci_upper=fitness_ci_upper,
        convergence_rate=convergence_rate,
        mean_generations_to_converge=mean_gens,
        std_generations_to_converge=std_gens,
        fitness_values=fitness_values,
        convergence_generations=convergence_gens,
        seeds=seeds
    )


def compare_conditions(
    condition_a: AggregatedCondition,
    condition_b: AggregatedCondition,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Perform statistical comparison between two conditions.

    Args:
        condition_a: First condition statistics
        condition_b: Second condition statistics
        alpha: Significance level (default: 0.05)

    Returns:
        ComparisonResult with statistical test results
    """
    analyzer = StatisticalAnalyzer()

    # Perform Welch's t-test and compute effect size
    result = analyzer.compare_fitness_distributions(
        evolved_scores=condition_a.fitness_values,
        baseline_scores=condition_b.fitness_values
    )

    # Compute improvement percentage
    mean_diff = condition_a.fitness_mean - condition_b.fitness_mean
    if condition_b.fitness_mean > 0:
        improvement_pct = (mean_diff / condition_b.fitness_mean) * 100
    else:
        improvement_pct = float('inf') if mean_diff > 0 else 0.0

    # Effect size interpretation
    d = result['effect_size']
    if abs(d) < 0.2:
        interpretation = "Negligible"
    elif abs(d) < 0.5:
        interpretation = "Small"
    elif abs(d) < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"

    return ComparisonResult(
        condition_a=condition_a.condition_name,
        condition_b=condition_b.condition_name,
        mean_difference=mean_diff,
        improvement_pct=improvement_pct,
        p_value=result['p_value'],
        effect_size=d,
        ci_lower=result['confidence_interval'][0],
        ci_upper=result['confidence_interval'][1],
        significant=result['p_value'] < alpha,
        interpretation=interpretation
    )


def print_summary_table(conditions: Dict[str, AggregatedCondition]):
    """Print formatted summary table of all conditions."""
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS SUMMARY")
    print("="*80 + "\n")

    print(f"{'Condition':<20} {'N':<6} {'Mean':<12} {'SD':<12} {'Median':<12} {'95% CI':<25}")
    print("-" * 80)

    for cond_name, cond in conditions.items():
        ci_str = f"[{cond.fitness_ci_lower:.1f}, {cond.fitness_ci_upper:.1f}]"
        print(f"{cond_name:<20} {cond.n_valid:<6} "
              f"{cond.fitness_mean:<12.1f} {cond.fitness_std:<12.1f} "
              f"{cond.fitness_median:<12.1f} {ci_str:<25}")

    print("\n" + "="*80)
    print("CONVERGENCE STATISTICS")
    print("="*80 + "\n")

    print(f"{'Condition':<20} {'Conv. Rate':<15} {'Mean Gens':<15} {'SD Gens':<15}")
    print("-" * 80)

    for cond_name, cond in conditions.items():
        conv_rate_str = f"{cond.convergence_rate*100:.1f}% ({int(cond.convergence_rate*cond.n_valid)}/{cond.n_valid})"
        mean_str = f"{cond.mean_generations_to_converge:.1f}" if cond.mean_generations_to_converge else "N/A"
        std_str = f"{cond.std_generations_to_converge:.1f}" if cond.std_generations_to_converge else "N/A"
        print(f"{cond_name:<20} {conv_rate_str:<15} {mean_str:<15} {std_str:<15}")

    print()


def print_comparison_table(comparisons: List[ComparisonResult]):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("STATISTICAL COMPARISONS")
    print("="*100 + "\n")

    print(f"{'Comparison':<30} {'Δ Mean':<12} {'Improve %':<12} {'p-value':<12} "
          f"{'Cohen\'s d':<12} {'Sig?':<8} {'Effect':<12}")
    print("-" * 100)

    for comp in comparisons:
        comparison_str = f"{comp.condition_a} vs {comp.condition_b}"
        sig_str = "***" if comp.p_value < 0.001 else "**" if comp.p_value < 0.01 else "*" if comp.significant else "ns"
        improve_str = f"+{comp.improvement_pct:.1f}%" if comp.improvement_pct != float('inf') else "+∞%"

        print(f"{comparison_str:<30} {comp.mean_difference:<12.1f} {improve_str:<12} "
              f"{comp.p_value:<12.4f} {comp.effect_size:<12.2f} {sig_str:<8} {comp.interpretation:<12}")

    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experimental results and perform statistical analysis"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result JSON files (default: results/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/aggregated_results.json"),
        help="Output file for aggregated results (default: results/aggregated_results.json)"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/statistical_summary.txt"),
        help="Output file for human-readable summary (default: results/statistical_summary.txt)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS AGGREGATION")
    print("="*80 + "\n")

    # Load results for each condition
    conditions = {}

    # Rule-based results
    rulebased_results = load_comparison_results(args.results_dir, "rulebased_run_*.json")
    if rulebased_results:
        conditions['Rule-Based'] = compute_aggregated_stats('Rule-Based', rulebased_results)

    # LLM results
    llm_results = load_comparison_results(args.results_dir, "llm_run_*.json")
    if llm_results:
        conditions['LLM-Guided'] = compute_aggregated_stats('LLM-Guided', llm_results)

    # Meta-learning results
    meta_results = load_comparison_results(args.results_dir, "metalearning_run_*.json")
    if meta_results:
        conditions['LLM+Meta'] = compute_aggregated_stats('LLM+Meta', meta_results)

    # Baseline validation results (Phase 1 pilot)
    baseline_results = load_comparison_results(args.results_dir, "baseline_validation.json")
    if baseline_results:
        conditions['Baseline-Pilot'] = compute_aggregated_stats('Baseline-Pilot', baseline_results)

    # LLM pilot results (Phase 1)
    llm_pilot_results = load_comparison_results(args.results_dir, "llm_pilot.json")
    if llm_pilot_results:
        conditions['LLM-Pilot'] = compute_aggregated_stats('LLM-Pilot', llm_pilot_results)

    if not conditions:
        print("❌ Error: No valid conditions found. Check that result files exist in the specified directory.")
        sys.exit(1)

    # Print summary tables
    print_summary_table(conditions)

    # Perform pairwise comparisons
    comparisons = []

    # Primary comparison: LLM vs Rule-Based
    if 'LLM-Guided' in conditions and 'Rule-Based' in conditions:
        comp = compare_conditions(conditions['LLM-Guided'], conditions['Rule-Based'])
        comparisons.append(comp)

    # Secondary comparison: LLM+Meta vs LLM
    if 'LLM+Meta' in conditions and 'LLM-Guided' in conditions:
        comp = compare_conditions(conditions['LLM+Meta'], conditions['LLM-Guided'])
        comparisons.append(comp)

    # Pilot comparisons
    if 'LLM-Pilot' in conditions and 'Baseline-Pilot' in conditions:
        comp = compare_conditions(conditions['LLM-Pilot'], conditions['Baseline-Pilot'])
        comparisons.append(comp)

    if comparisons:
        print_comparison_table(comparisons)

    # Save aggregated results to JSON
    output_data = {
        'conditions': {name: asdict(cond) for name, cond in conditions.items()},
        'comparisons': [asdict(comp) for comp in comparisons]
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"💾 Aggregated results saved to: {args.output}")

    # Save human-readable summary
    with open(args.summary, 'w') as f:
        # Redirect print to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f

        print_summary_table(conditions)
        if comparisons:
            print_comparison_table(comparisons)

        sys.stdout = old_stdout

    print(f"📄 Statistical summary saved to: {args.summary}")

    print("\n✅ Aggregation complete!\n")


if __name__ == "__main__":
    main()
