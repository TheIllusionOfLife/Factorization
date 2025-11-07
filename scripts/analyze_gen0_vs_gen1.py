#!/usr/bin/env python3
"""Analyze Gen 0 vs Gen 1 to evaluate LLM's impact on first generation.

This script examines the critical transition from random initialization (Gen 0)
to LLM-guided mutations (Gen 1) to determine if:
1. Offspring are sufficiently diversified
2. Fitness scores improve from Gen 0 to Gen 1
3. LLM provides meaningful mutations vs random search

Usage:
    python scripts/analyze_gen0_vs_gen1.py results/c2_validation/c2_elite_pilot_seed9015.json
    python scripts/analyze_gen0_vs_gen1.py results/c2_validation/*.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def load_experiment(filepath: str) -> Dict:
    """Load experiment JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_generation(data: Dict, gen_num: int) -> Dict:
    """Extract specific generation from history."""
    gen_history = data.get("generation_history", [])
    if gen_num >= len(gen_history):
        return None
    return gen_history[gen_num]


def calculate_diversity(strategies: List[Dict]) -> Dict[str, float]:
    """Calculate diversity metrics for a generation's strategies.

    Returns:
        Dict with diversity metrics:
        - unique_powers: Count of unique power values
        - unique_smoothness: Count of unique smoothness_bound values
        - unique_min_hits: Count of unique min_small_prime_hits values
        - avg_filter_count: Average number of modulus filters
    """
    powers = [s["strategy"]["power"] for s in strategies]
    smoothness = [s["strategy"]["smoothness_bound"] for s in strategies]
    min_hits = [s["strategy"]["min_small_prime_hits"] for s in strategies]
    filter_counts = [len(s["strategy"]["modulus_filters"]) for s in strategies]

    return {
        "unique_powers": len(set(powers)),
        "unique_smoothness": len(set(smoothness)),
        "unique_min_hits": len(set(min_hits)),
        "avg_filter_count": sum(filter_counts) / len(filter_counts)
        if filter_counts
        else 0,
        "power_distribution": {p: powers.count(p) for p in set(powers)},
    }


def calculate_fitness_stats(strategies: List[Dict]) -> Dict[str, float]:
    """Calculate fitness statistics for a generation."""
    fitnesses = [s["fitness"] for s in strategies]
    if not fitnesses:
        return {"mean": 0, "min": 0, "max": 0, "median": 0}

    sorted_fitnesses = sorted(fitnesses)
    n = len(sorted_fitnesses)
    median = (
        sorted_fitnesses[n // 2]
        if n % 2 == 1
        else (sorted_fitnesses[n // 2 - 1] + sorted_fitnesses[n // 2]) / 2
    )

    return {
        "mean": sum(fitnesses) / len(fitnesses),
        "min": min(fitnesses),
        "max": max(fitnesses),
        "median": median,
        "nonzero_count": sum(1 for f in fitnesses if f > 0),
    }


def compare_generations(gen0: Dict, gen1: Dict) -> Dict:
    """Compare Gen 0 vs Gen 1 to assess LLM impact."""
    gen0_strategies = gen0["all_strategies"]
    gen1_strategies = gen1["all_strategies"]

    gen0_fitness = calculate_fitness_stats(gen0_strategies)
    gen1_fitness = calculate_fitness_stats(gen1_strategies)

    gen0_diversity = calculate_diversity(gen0_strategies)
    gen1_diversity = calculate_diversity(gen1_strategies)

    # Calculate improvement metrics
    fitness_improvement = {
        "mean_improvement": gen1_fitness["mean"] - gen0_fitness["mean"],
        "mean_improvement_pct": ((gen1_fitness["mean"] / gen0_fitness["mean"]) - 1)
        * 100
        if gen0_fitness["mean"] > 0
        else float("inf"),
        "max_improvement": gen1_fitness["max"] - gen0_fitness["max"],
        "nonzero_delta": gen1_fitness["nonzero_count"] - gen0_fitness["nonzero_count"],
    }

    return {
        "gen0_fitness": gen0_fitness,
        "gen1_fitness": gen1_fitness,
        "gen0_diversity": gen0_diversity,
        "gen1_diversity": gen1_diversity,
        "fitness_improvement": fitness_improvement,
    }


def print_analysis(filepath: str, analysis: Dict):
    """Print human-readable analysis."""
    print(f"\n{'=' * 80}")
    print(f"Analysis: {Path(filepath).name}")
    print(f"{'=' * 80}")

    gen0_fit = analysis["gen0_fitness"]
    gen1_fit = analysis["gen1_fitness"]
    improvement = analysis["fitness_improvement"]

    print("\n📊 FITNESS COMPARISON")
    print(
        f"Gen 0: mean={gen0_fit['mean']:.0f}, max={gen0_fit['max']:.0f}, nonzero={gen0_fit['nonzero_count']}/{20}"
    )
    print(
        f"Gen 1: mean={gen1_fit['mean']:.0f}, max={gen1_fit['max']:.0f}, nonzero={gen1_fit['nonzero_count']}/{20}"
    )

    mean_imp = improvement["mean_improvement"]
    mean_imp_pct = improvement["mean_improvement_pct"]
    max_imp = improvement["max_improvement"]

    print(
        f"\n{'✅' if mean_imp > 0 else '❌'} Mean fitness change: {mean_imp:+.0f} ({mean_imp_pct:+.1f}%)"
    )
    print(f"{'✅' if max_imp > 0 else '❌'} Max fitness change: {max_imp:+.0f}")
    print(
        f"{'✅' if improvement['nonzero_delta'] >= 0 else '❌'} Nonzero strategies: {improvement['nonzero_delta']:+d}"
    )

    gen0_div = analysis["gen0_diversity"]
    gen1_div = analysis["gen1_diversity"]

    print("\n🔀 DIVERSITY ANALYSIS")
    print(
        f"Gen 0: powers={gen0_div['unique_powers']}, smoothness={gen0_div['unique_smoothness']}, "
        f"min_hits={gen0_div['unique_min_hits']}, avg_filters={gen0_div['avg_filter_count']:.1f}"
    )
    print(
        f"Gen 1: powers={gen1_div['unique_powers']}, smoothness={gen1_div['unique_smoothness']}, "
        f"min_hits={gen1_div['unique_min_hits']}, avg_filters={gen1_div['avg_filter_count']:.1f}"
    )

    print(f"\nGen 0 power distribution: {gen0_div['power_distribution']}")
    print(f"Gen 1 power distribution: {gen1_div['power_distribution']}")

    # Overall assessment
    print("\n🎯 ASSESSMENT")
    is_improved = mean_imp > 0 and max_imp > 0
    is_diverse = gen1_div["unique_powers"] >= 2 and gen1_div["unique_smoothness"] >= 2

    if is_improved and is_diverse:
        print("✅ LLM effectively improves fitness AND maintains diversity")
    elif is_improved:
        print("⚠️  LLM improves fitness but diversity may be limited")
    elif is_diverse:
        print("⚠️  LLM maintains diversity but fitness does not improve")
    else:
        print("❌ LLM shows limited impact on both fitness and diversity")


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/analyze_gen0_vs_gen1.py <json_file> [<json_file> ...]"
        )
        print("\nExample:")
        print(
            "  python scripts/analyze_gen0_vs_gen1.py results/c2_validation/c2_elite_pilot_seed9015.json"
        )
        print("  python scripts/analyze_gen0_vs_gen1.py results/c2_validation/*.json")
        sys.exit(1)

    filepaths = sys.argv[1:]

    # Aggregate results across all experiments
    all_analyses = []

    for filepath in filepaths:
        if not Path(filepath).exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        try:
            data = load_experiment(filepath)

            # Check if generation_history exists
            if "generation_history" not in data:
                print(f"⚠️  No generation_history in {filepath} (old format)")
                continue

            gen0 = extract_generation(data, 0)
            gen1 = extract_generation(data, 1)

            if not gen0 or not gen1:
                print(f"⚠️  Missing Gen 0 or Gen 1 in {filepath}")
                continue

            analysis = compare_generations(gen0, gen1)
            print_analysis(filepath, analysis)
            all_analyses.append((filepath, analysis))

        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            continue

    # Summary across all experiments
    if len(all_analyses) > 1:
        print(f"\n{'=' * 80}")
        print(f"AGGREGATE SUMMARY ({len(all_analyses)} experiments)")
        print(f"{'=' * 80}")

        mean_improvements = [
            a["fitness_improvement"]["mean_improvement"] for _, a in all_analyses
        ]
        max_improvements = [
            a["fitness_improvement"]["max_improvement"] for _, a in all_analyses
        ]

        positive_mean = sum(1 for x in mean_improvements if x > 0)
        positive_max = sum(1 for x in max_improvements if x > 0)

        print("\n📊 Fitness improvements:")
        print(
            f"  Mean improvement > 0: {positive_mean}/{len(all_analyses)} experiments ({100 * positive_mean / len(all_analyses):.0f}%)"
        )
        print(
            f"  Max improvement > 0: {positive_max}/{len(all_analyses)} experiments ({100 * positive_max / len(all_analyses):.0f}%)"
        )

        avg_mean_imp = sum(mean_improvements) / len(mean_improvements)
        avg_max_imp = sum(max_improvements) / len(max_improvements)
        print(f"  Average mean improvement: {avg_mean_imp:+.0f}")
        print(f"  Average max improvement: {avg_max_imp:+.0f}")

        # Diversity summary
        unique_powers_g0 = [
            a["gen0_diversity"]["unique_powers"] for _, a in all_analyses
        ]
        unique_powers_g1 = [
            a["gen1_diversity"]["unique_powers"] for _, a in all_analyses
        ]

        print("\n🔀 Diversity:")
        print(
            f"  Gen 0 avg unique powers: {sum(unique_powers_g0) / len(unique_powers_g0):.1f}"
        )
        print(
            f"  Gen 1 avg unique powers: {sum(unique_powers_g1) / len(unique_powers_g1):.1f}"
        )


if __name__ == "__main__":
    main()
