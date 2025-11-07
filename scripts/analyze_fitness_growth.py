#!/usr/bin/env python3
"""Analyze generation-by-generation fitness growth across experiments.

This script tracks fitness evolution from Gen 0 through Gen 5 (or all 20 generations)
to understand how LLM-guided mutations affect learning curves over time.

Usage:
    python scripts/analyze_fitness_growth.py results/c2_validation/c2_gen_analysis_seed*.json
    python scripts/analyze_fitness_growth.py results/c1_validation/*.json --max-gen 5
    python scripts/analyze_fitness_growth.py results/c2_validation/*.json --compare results/c1_validation/*.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_experiment(filepath: str) -> Dict:
    """Load experiment JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_fitness_trajectory(
    data: Dict, max_gen: int = 20
) -> Tuple[List[float], List[float]]:
    """Extract mean and max fitness per generation.

    Returns:
        Tuple of (mean_fitnesses, max_fitnesses) lists
    """
    gen_history = data.get("generation_history", [])
    if not gen_history:
        # Fallback: use best_fitness as final generation
        print(
            f"⚠️  No generation_history in {Path(data.get('random_seed', 'unknown')).name} (old format)"
        )
        return [], []

    mean_fitnesses = []
    max_fitnesses = []

    for gen in gen_history[: min(max_gen, len(gen_history))]:
        strategies = gen.get("all_strategies", [])
        if not strategies:
            continue

        fitnesses = [s["fitness"] for s in strategies]
        mean_fitnesses.append(sum(fitnesses) / len(fitnesses))
        max_fitnesses.append(max(fitnesses))

    return mean_fitnesses, max_fitnesses


def calculate_growth_rate(trajectory: List[float]) -> Dict[str, float]:
    """Calculate growth metrics for a fitness trajectory.

    Returns:
        Dict with growth rate metrics:
        - total_growth: final / initial
        - avg_growth_per_gen: geometric mean of per-generation growth
        - max_jump: largest single-generation improvement
    """
    if len(trajectory) < 2:
        return {
            "total_growth": 1.0,
            "avg_growth_per_gen": 1.0,
            "max_jump": 0.0,
            "max_jump_gen": -1,
        }

    # Total growth (final / initial, handle zero)
    initial = trajectory[0] if trajectory[0] > 0 else 1
    final = trajectory[-1]
    total_growth = final / initial

    # Per-generation growth rates
    growth_rates = []
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1] if trajectory[i - 1] > 0 else 1
        curr = trajectory[i]
        growth_rates.append(curr / prev)

    # Geometric mean of growth rates: (∏ rate)^(1/n)
    if growth_rates:
        product = 1.0
        for rate in growth_rates:
            product *= rate
        geometric_mean = product ** (1 / len(growth_rates))
    else:
        geometric_mean = 1.0

    # Max single-generation jump
    jumps = [trajectory[i] - trajectory[i - 1] for i in range(1, len(trajectory))]
    max_jump = max(jumps) if jumps else 0.0
    max_jump_gen = jumps.index(max_jump) + 1 if jumps else -1

    return {
        "total_growth": total_growth,
        "avg_growth_per_gen": geometric_mean,
        "max_jump": max_jump,
        "max_jump_gen": max_jump_gen,
    }


def aggregate_trajectories(
    experiments: List[Tuple[str, List[float], List[float]]],
) -> Tuple[List[float], List[float]]:
    """Aggregate multiple fitness trajectories into mean trajectory.

    Returns:
        Tuple of (mean_trajectory, max_trajectory) where each is averaged across experiments
    """
    if not experiments:
        return [], []

    # Find min length to align all trajectories
    min_len = min(len(exp[1]) for exp in experiments)

    mean_trajectory = []
    max_trajectory = []

    for gen in range(min_len):
        mean_values = [exp[1][gen] for exp in experiments]
        max_values = [exp[2][gen] for exp in experiments]

        mean_trajectory.append(sum(mean_values) / len(mean_values))
        max_trajectory.append(sum(max_values) / len(max_values))

    return mean_trajectory, max_trajectory


def print_trajectory(
    name: str, mean_traj: List[float], max_traj: List[float], max_gen: int
):
    """Print fitness trajectory in tabular format."""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    print(
        f"{'Gen':<5} {'Mean Fitness':>15} {'Max Fitness':>15} {'Mean Growth':>12} {'Max Growth':>12}"
    )
    print(f"{'-' * 80}")

    for gen in range(min(max_gen, len(mean_traj))):
        mean_val = mean_traj[gen]
        max_val = max_traj[gen]

        # Growth since previous generation
        if gen > 0:
            mean_growth = (
                (mean_val / mean_traj[gen - 1] - 1) * 100
                if mean_traj[gen - 1] > 0
                else float("inf")
            )
            max_growth = (
                (max_val / max_traj[gen - 1] - 1) * 100
                if max_traj[gen - 1] > 0
                else float("inf")
            )
        else:
            mean_growth = 0.0
            max_growth = 0.0

        growth_str_mean = (
            f"{mean_growth:+.1f}%" if mean_growth != float("inf") else "+inf%"
        )
        growth_str_max = (
            f"{max_growth:+.1f}%" if max_growth != float("inf") else "+inf%"
        )

        print(
            f"{gen:<5} {mean_val:>15,.0f} {max_val:>15,.0f} {growth_str_mean:>12} {growth_str_max:>12}"
        )

    # Summary statistics
    growth_metrics_mean = calculate_growth_rate(mean_traj[:max_gen])
    growth_metrics_max = calculate_growth_rate(max_traj[:max_gen])

    print(f"\n📊 Summary (Gen 0 → Gen {min(max_gen, len(mean_traj)) - 1}):")
    print(
        f"  Mean fitness: {mean_traj[0]:,.0f} → {mean_traj[min(max_gen, len(mean_traj)) - 1]:,.0f} "
        f"({growth_metrics_mean['total_growth']:.2f}x total)"
    )
    print(
        f"  Max fitness: {max_traj[0]:,.0f} → {max_traj[min(max_gen, len(max_traj)) - 1]:,.0f} "
        f"({growth_metrics_max['total_growth']:.2f}x total)"
    )
    print(
        f"  Largest mean jump: +{growth_metrics_mean['max_jump']:,.0f} "
        f"(Gen {growth_metrics_mean['max_jump_gen'] - 1} → {growth_metrics_mean['max_jump_gen']})"
    )
    print(
        f"  Largest max jump: +{growth_metrics_max['max_jump']:,.0f} "
        f"(Gen {growth_metrics_max['max_jump_gen'] - 1} → {growth_metrics_max['max_jump_gen']})"
    )


def compare_modes(
    mode1_name: str,
    mode1_traj: Tuple[List[float], List[float]],
    mode2_name: str,
    mode2_traj: Tuple[List[float], List[float]],
    max_gen: int,
):
    """Compare two modes side-by-side."""
    mean1, max1 = mode1_traj
    mean2, max2 = mode2_traj

    min_len = min(len(mean1), len(mean2), max_gen)

    print(f"\n{'=' * 120}")
    print(f"COMPARISON: {mode1_name} vs {mode2_name}")
    print(f"{'=' * 120}")
    print(
        f"{'Gen':<5} "
        f"{f'{mode1_name} Mean':>18} {f'{mode2_name} Mean':>18} {'Δ Mean':>12} "
        f"{f'{mode1_name} Max':>18} {f'{mode2_name} Max':>18} {'Δ Max':>12}"
    )
    print(f"{'-' * 120}")

    for gen in range(min_len):
        mean1_val = mean1[gen]
        mean2_val = mean2[gen]
        max1_val = max1[gen]
        max2_val = max2[gen]

        delta_mean = mean2_val - mean1_val
        delta_max = max2_val - max1_val

        print(
            f"{gen:<5} "
            f"{mean1_val:>18,.0f} {mean2_val:>18,.0f} {delta_mean:+12,.0f} "
            f"{max1_val:>18,.0f} {max2_val:>18,.0f} {delta_max:+12,.0f}"
        )

    # Final comparison
    print(f"\n🔬 Final Generation ({min_len - 1}) Analysis:")
    final_mean_ratio = (
        mean2[min_len - 1] / mean1[min_len - 1]
        if mean1[min_len - 1] > 0
        else float("inf")
    )
    final_max_ratio = (
        max2[min_len - 1] / max1[min_len - 1] if max1[min_len - 1] > 0 else float("inf")
    )

    print(
        f"  {mode2_name} mean fitness: {final_mean_ratio:.2f}x {mode1_name} "
        f"({'better' if final_mean_ratio > 1 else 'worse'})"
    )
    print(
        f"  {mode2_name} max fitness: {final_max_ratio:.2f}x {mode1_name} "
        f"({'better' if final_max_ratio > 1 else 'worse'})"
    )


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze generation-by-generation fitness growth"
    )
    parser.add_argument("files", nargs="+", help="Experiment JSON files to analyze")
    parser.add_argument(
        "--max-gen",
        type=int,
        default=20,
        help="Maximum generation to analyze (default: 20)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare with another set of experiments",
    )
    parser.add_argument(
        "--mode1-name",
        default="Mode 1",
        help="Name for first mode (default: Mode 1)",
    )
    parser.add_argument(
        "--mode2-name",
        default="Mode 2",
        help="Name for second mode (default: Mode 2)",
    )

    args = parser.parse_args()

    # Load and process primary experiments
    experiments = []
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        try:
            data = load_experiment(filepath)
            mean_traj, max_traj = extract_fitness_trajectory(data, args.max_gen)

            if mean_traj:
                experiments.append((filepath, mean_traj, max_traj))
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            continue

    if not experiments:
        print("❌ No valid experiments found")
        sys.exit(1)

    # Print individual trajectories
    for filepath, mean_traj, max_traj in experiments:
        print_trajectory(Path(filepath).name, mean_traj, max_traj, args.max_gen)

    # Aggregate and print average trajectory
    if len(experiments) > 1:
        mean_agg, max_agg = aggregate_trajectories(experiments)
        print_trajectory(
            f"AGGREGATE ({len(experiments)} experiments)",
            mean_agg,
            max_agg,
            args.max_gen,
        )

    # Comparison mode
    if args.compare:
        compare_experiments = []
        for filepath in args.compare:
            if not Path(filepath).exists():
                print(f"⚠️  Comparison file not found: {filepath}")
                continue

            try:
                data = load_experiment(filepath)
                mean_traj, max_traj = extract_fitness_trajectory(data, args.max_gen)

                if mean_traj:
                    compare_experiments.append((filepath, mean_traj, max_traj))
            except Exception as e:
                print(f"❌ Error processing comparison file {filepath}: {e}")
                continue

        if compare_experiments:
            # Aggregate comparison trajectories
            mean_compare, max_compare = aggregate_trajectories(compare_experiments)
            mean_primary, max_primary = aggregate_trajectories(experiments)

            compare_modes(
                args.mode1_name,
                (mean_primary, max_primary),
                args.mode2_name,
                (mean_compare, max_compare),
                args.max_gen,
            )


if __name__ == "__main__":
    main()
