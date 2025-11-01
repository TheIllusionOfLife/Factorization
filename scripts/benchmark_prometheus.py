#!/usr/bin/env python3
"""
Benchmark script for Prometheus Phase 1 infrastructure validation.

Measures:
- Message passing overhead
- Memory efficiency across all 4 modes
- Execution time per generation
- Communication efficiency metrics
"""

import argparse
import json
import sys
import time
import traceback
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.prometheus.experiment import PrometheusExperiment

# Seed offsets ensure independent RNG streams per mode for fair comparison
# Each mode gets unique offset to prevent correlation in random number generation
MODE_SEED_OFFSETS = {
    "collaborative": 0,
    "search_only": 1000,
    "eval_only": 2000,
    "rulebased": 3000,
}


def measure_experiment(
    mode: str,
    generations: int,
    population: int,
    duration: float,
    seed: int,
    track_memory: bool = True,
) -> Dict[str, Any]:
    """Run a single experiment and measure performance metrics."""

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {mode} mode")
    print(f"Config: {generations} gen × {population} pop × {duration}s eval")
    print(f"{'=' * 60}")

    # Start memory tracking
    if track_memory:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        mem_start = tracemalloc.get_traced_memory()[0]

    # Measure wall clock time
    time_start = time.perf_counter()

    # Create config
    config = Config(
        api_key="dummy_key_for_benchmarks",  # Not used when llm_enabled=False
        evaluation_duration=duration,
        llm_enabled=False,  # Rule-based for Phase 1 benchmarks
        prometheus_enabled=True,
        prometheus_mode=mode,
    )

    # Run experiment
    experiment = PrometheusExperiment(
        config=config,
        target_number=961730063,
        random_seed=seed,
    )

    # Run based on mode
    if mode == "collaborative":
        best_fitness, best_strategy, comm_stats = (
            experiment.run_collaborative_evolution(
                generations=generations,
                population_size=population,
            )
        )
        results = {
            "best_fitness": best_fitness,
            "total_messages": comm_stats.get("total_messages", 0),
        }
    else:  # search_only, eval_only, rulebased
        best_fitness, best_strategy = experiment.run_independent_baseline(
            agent_type=mode,
            generations=generations,
            population_size=population,
        )
        results = {
            "best_fitness": best_fitness,
            "total_messages": 0,
        }

    # Measure end time
    time_end = time.perf_counter()
    elapsed_time = time_end - time_start

    # Measure memory
    if track_memory:
        mem_end, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_used = mem_end - mem_start
    else:
        mem_used = 0
        mem_peak = 0

    # Calculate metrics
    metrics = {
        "mode": mode,
        "config": {
            "generations": generations,
            "population": population,
            "evaluation_duration": duration,
            "seed": seed,
        },
        "performance": {
            "total_time_seconds": round(elapsed_time, 3),
            "time_per_generation": round(elapsed_time / generations, 3)
            if generations > 0
            else 0,
            "time_per_evaluation": round(elapsed_time / (generations * population), 3)
            if (generations * population) > 0
            else 0,
            "memory_delta_mb": round(mem_used / 1024 / 1024, 2),
            "memory_peak_mb": round(mem_peak / 1024 / 1024, 2),
        },
        "results": {
            "best_fitness": results["best_fitness"],
            "total_messages": results.get("total_messages", 0),
            "emergence_factor": results.get("emergence_factor"),
            "synergy_score": results.get("synergy_score"),
            "communication_efficiency": results.get("communication_efficiency"),
        },
    }

    # Calculate message overhead if applicable
    if results.get("total_messages", 0) > 0:
        metrics["performance"]["time_per_message_ms"] = round(
            (elapsed_time / results["total_messages"]) * 1000, 2
        )

    # Print summary
    print("\n📊 Performance Summary:")
    print(f"  Total time: {metrics['performance']['total_time_seconds']}s")
    print(f"  Time per generation: {metrics['performance']['time_per_generation']}s")
    print(f"  Time per evaluation: {metrics['performance']['time_per_evaluation']}s")
    print(f"  Memory delta: {metrics['performance']['memory_delta_mb']} MB")
    print(f"  Memory peak: {metrics['performance']['memory_peak_mb']} MB")

    if results.get("total_messages", 0) > 0:
        print(f"  Messages sent: {results['total_messages']}")
        print(f"  Time per message: {metrics['performance']['time_per_message_ms']} ms")

    print("\n🎯 Results:")
    print(f"  Best fitness: {metrics['results']['best_fitness']:,}")
    if metrics["results"]["emergence_factor"]:
        print(f"  Emergence factor: {metrics['results']['emergence_factor']:.3f}")

    return metrics


def run_benchmark_suite(
    modes: List[str],
    generations: int,
    population: int,
    duration: float,
    seed: int,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run benchmarks for all specified modes."""

    print("\n" + "=" * 60)
    print("🚀 Prometheus Phase 1 Performance Benchmark Suite")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Modes: {', '.join(modes)}")
    print(f"  Generations: {generations}")
    print(f"  Population: {population}")
    print(f"  Evaluation duration: {duration}s")
    print(f"  Random seed: {seed}")

    all_results = {
        "benchmark_config": {
            "modes": modes,
            "generations": generations,
            "population": population,
            "evaluation_duration": duration,
            "seed": seed,
        },
        "mode_results": {},
    }

    # Run benchmarks for each mode
    for mode in modes:
        try:
            mode_metrics = measure_experiment(
                mode=mode,
                generations=generations,
                population=population,
                duration=duration,
                seed=seed + MODE_SEED_OFFSETS[mode],
            )
            all_results["mode_results"][mode] = mode_metrics
        except Exception as e:
            print(f"\n❌ Error benchmarking {mode} mode: {e}")
            traceback.print_exc()
            all_results["mode_results"][mode] = {"error": str(e)}

    # Print comparison summary
    print("\n" + "=" * 60)
    print("📈 Benchmark Comparison Summary")
    print("=" * 60)

    print(
        f"\n{'Mode':<15} {'Time (s)':<12} {'Mem (MB)':<12} {'Fitness':<12} {'Msgs':<8}"
    )
    print("-" * 65)

    for mode, metrics in all_results["mode_results"].items():
        if "error" in metrics:
            print(f"{mode:<15} ERROR")
            continue

        perf = metrics["performance"]
        results = metrics["results"]

        print(
            f"{mode:<15} "
            f"{perf['total_time_seconds']:<12.2f} "
            f"{perf['memory_peak_mb']:<12.2f} "
            f"{results['best_fitness']:<12,} "
            f"{results['total_messages']:<8}"
        )

    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Prometheus Phase 1 infrastructure"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["collaborative", "search_only", "eval_only", "rulebased"],
        choices=["collaborative", "search_only", "eval_only", "rulebased"],
        help="Modes to benchmark (default: all)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations (default: 10)",
    )
    parser.add_argument(
        "--population", type=int, default=10, help="Population size (default: 10)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.5,
        help="Evaluation duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="Random seed (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmarks/prometheus_phase1_validation.json",
        help="Output file for results (default: results/benchmarks/prometheus_phase1_validation.json)",
    )

    args = parser.parse_args()

    # Run benchmark suite
    results = run_benchmark_suite(
        modes=args.modes,
        generations=args.generations,
        population=args.population,
        duration=args.duration,
        seed=args.seed,
        output_file=args.output,
    )

    # Exit with error if any benchmark failed
    for _mode, metrics in results["mode_results"].items():
        if "error" in metrics:
            print("\n❌ Benchmark suite completed with errors")
            sys.exit(1)

    print("\n✅ Benchmark suite completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
