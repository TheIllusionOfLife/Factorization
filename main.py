"""Main CLI entry point for evolutionary GNFS strategy optimizer."""

import argparse
import json
import sys
from pathlib import Path

from src.crucible import FactorizationCrucible
from src.evolution import EvolutionaryEngine


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary GNFS strategy optimizer with optional LLM integration"
    )
    parser.add_argument(
        "--number",
        type=int,
        default=961730063,
        help="Number to factor (default: 961730063)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations to evolve (default: 5)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=10,
        help="Population size per generation (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Evaluation duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-guided mutations (requires GEMINI_API_KEY in .env)",
    )
    parser.add_argument(
        "--export-metrics",
        type=str,
        metavar="PATH",
        help="Export detailed metrics to JSON file (e.g., metrics/run_001.json)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.3,
        metavar="RATE",
        help="Crossover rate: fraction of offspring from two parents (default: 0.3)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.5,
        metavar="RATE",
        help="Mutation rate: fraction of offspring from single parent (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="SEED",
        help="Random seed for reproducible runs (e.g., 42). Omit for non-deterministic behavior.",
    )

    # Meta-learning arguments
    parser.add_argument(
        "--meta-learning",
        action="store_true",
        help="Enable meta-learning: adapt operator rates based on performance",
    )
    parser.add_argument(
        "--adaptation-window",
        type=int,
        default=5,
        metavar="N",
        help="Generations to consider for rate adaptation (default: 5)",
    )

    # Comparison mode arguments
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Run comparison against baseline strategies with statistical analysis",
    )
    parser.add_argument(
        "--num-comparison-runs",
        type=int,
        default=5,
        metavar="N",
        help="Number of independent runs for comparison (default: 5)",
    )
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=5,
        metavar="N",
        help="Generation window for convergence detection (default: 5)",
    )
    parser.add_argument(
        "--export-comparison",
        type=str,
        metavar="PATH",
        help="Export comparison results to JSON file",
    )

    args = parser.parse_args()

    # Validate reproduction rates
    if args.crossover_rate < 0 or args.crossover_rate > 1:
        print(
            f"❌ ERROR: crossover-rate must be between 0 and 1 (got {args.crossover_rate})"
        )
        sys.exit(1)
    if args.mutation_rate < 0 or args.mutation_rate > 1:
        print(
            f"❌ ERROR: mutation-rate must be between 0 and 1 (got {args.mutation_rate})"
        )
        sys.exit(1)
    if args.crossover_rate + args.mutation_rate > 1.0:
        print("❌ ERROR: crossover-rate + mutation-rate must be <= 1.0")
        print(
            f"   (got {args.crossover_rate} + {args.mutation_rate} = {args.crossover_rate + args.mutation_rate})"
        )
        sys.exit(1)

    # Validate core loop sizes
    if args.generations < 1:
        print(f"❌ ERROR: generations must be >= 1 (got {args.generations})")
        sys.exit(1)
    if args.population < 1:
        print(f"❌ ERROR: population must be >= 1 (got {args.population})")
        sys.exit(1)
    if args.compare_baseline and args.num_comparison_runs < 1:
        print(
            f"❌ ERROR: num-comparison-runs must be >= 1 (got {args.num_comparison_runs})"
        )
        sys.exit(1)

    # Initialize LLM provider if requested
    llm_provider = None
    if args.llm:
        try:
            from src.comparison import print_llm_summary
            from src.config import load_config
            from src.llm.gemini import GeminiProvider

            config = load_config()
            if not config.api_key:
                print("❌ ERROR: GEMINI_API_KEY not set in .env file")
                print("Please create .env file with your API key (see .env.example)")
                sys.exit(1)

            llm_provider = GeminiProvider(config.api_key, config)
            print("✅ LLM mode enabled (Gemini 2.5 Flash Lite)")
            print(f"   Max API calls: {config.max_llm_calls}")
        except ValueError as e:
            print(f"❌ ERROR: {e}")
            print("Hint: set GEMINI_API_KEY in .env or run without --llm")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ ERROR: Missing dependencies for LLM mode: {e}")
            print("Please run: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("📊 Rule-based mode (no LLM)")

    # Create crucible
    crucible = FactorizationCrucible(args.number)

    # Comparison mode vs normal evolution mode
    if args.compare_baseline:
        # Import optional comparison dependencies only when needed
        from src.comparison import ComparisonEngine

        # Comparison mode: Run vs baselines with statistical analysis
        print(f"\n🎯 Target number: {args.number}")
        print(f"🧬 Generations: {args.generations}, Population: {args.population}")
        print(f"⏱️  Evaluation duration: {args.duration}s per strategy")
        print(f"📊 Comparison runs: {args.num_comparison_runs}")
        print(f"🔍 Convergence window: {args.convergence_window} generations")
        if args.seed is not None:
            print(f"🎲 Base seed: {args.seed} (reproducible runs)")
        print()

        comparison_engine = ComparisonEngine(
            crucible=crucible,
            num_runs=args.num_comparison_runs,
            max_generations=args.generations,
            population_size=args.population,
            evaluation_duration=args.duration,
            convergence_window=args.convergence_window,
            llm_provider=llm_provider,
        )

        runs = comparison_engine.run_comparison(base_seed=args.seed)
        analysis = comparison_engine.analyze_results(runs)

        # Print results
        print("\n" + "=" * 60)
        print("STATISTICAL COMPARISON RESULTS")
        print("=" * 60)

        for baseline_name, result in analysis["comparison_results"].items():
            improvement_pct = (
                (result.evolved_mean / result.baseline_mean - 1) * 100
                if result.baseline_mean > 0
                else float("inf")
            )
            print(f"\n{baseline_name.upper()} BASELINE:")
            print(f"  Evolved mean:  {result.evolved_mean:.1f}")
            print(f"  Baseline mean: {result.baseline_mean:.1f}")
            print(f"  Improvement:   {improvement_pct:+.1f}%")
            print(
                f"  p-value:       {result.p_value:.4f} {'***' if result.is_significant else '(not significant)'}"
            )
            print(f"  Effect size:   {result.effect_size:.2f} (Cohen's d)")
            print(
                f"  95% CI:        [{result.confidence_interval[0]:.1f}, {result.confidence_interval[1]:.1f}]"
            )

        # Convergence stats
        conv_stats = analysis["convergence_stats"]
        print("\nCONVERGENCE STATISTICS:")
        print(
            f"  Convergence rate: {conv_stats['convergence_rate']:.0%} ({int(conv_stats['convergence_rate'] * args.num_comparison_runs)}/{args.num_comparison_runs} runs)"
        )
        if conv_stats["mean"] is not None:
            print(
                f"  Mean generations: {conv_stats['mean']:.1f} ± {conv_stats['std']:.1f}"
            )

        # Export if requested
        if args.export_comparison:
            data = {
                "target_number": args.number,
                "num_runs": args.num_comparison_runs,
                "max_generations": args.generations,
                "population_size": args.population,
                "evaluation_duration": args.duration,
                "base_seed": args.seed,
                "runs": [
                    {
                        "evolved_fitness": run.evolved_fitness,
                        "baseline_fitness": run.baseline_fitness,
                        "generations_to_convergence": run.generations_to_convergence,
                        "random_seed": run.random_seed,
                    }
                    for run in runs
                ],
                "analysis": {
                    "comparison_results": {
                        name: {
                            "evolved_mean": float(result.evolved_mean),
                            "baseline_mean": float(result.baseline_mean),
                            "p_value": float(result.p_value),
                            "effect_size": float(result.effect_size),
                            "is_significant": bool(result.is_significant),
                            "confidence_interval": [
                                float(result.confidence_interval[0]),
                                float(result.confidence_interval[1]),
                            ],
                        }
                        for name, result in analysis["comparison_results"].items()
                    },
                    "convergence_stats": conv_stats,
                },
            }

            output_file = Path(args.export_comparison)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\n📁 Comparison results exported to: {args.export_comparison}")

        # Display LLM cost summary if used
        if llm_provider:
            print_llm_summary(llm_provider)

    else:
        # Normal evolution mode
        engine = EvolutionaryEngine(
            crucible,
            population_size=args.population,
            llm_provider=llm_provider,
            evaluation_duration=args.duration,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            random_seed=args.seed,
            enable_meta_learning=args.meta_learning,
            adaptation_window=args.adaptation_window,
        )

        print(f"\n🎯 Target number: {args.number}")
        print(f"🧬 Generations: {args.generations}, Population: {args.population}")
        print(f"⏱️  Evaluation duration: {args.duration}s per strategy")
        if args.meta_learning:
            print(
                f"🔀 Reproduction: {args.crossover_rate:.0%} crossover (initial), {args.mutation_rate:.0%} mutation (initial), {engine.random_rate:.0%} random (initial)"
            )
            print(
                f"🧠 Meta-learning enabled: Rates will adapt based on performance (window={args.adaptation_window})"
            )
        else:
            print(
                f"🔀 Reproduction: {args.crossover_rate:.0%} crossover, {args.mutation_rate:.0%} mutation, {engine.random_rate:.0%} random"
            )
        if args.seed is not None:
            print(f"🎲 Random seed: {args.seed} (reproducible run)")
        print()

        engine.initialize_population()

        for _ in range(args.generations):
            _best_fitness, _best_strategy = engine.run_evolutionary_cycle()

        # Display LLM cost summary if used
        if llm_provider:
            print_llm_summary(llm_provider)

        # Export metrics if requested
        if args.export_metrics:
            print()
            engine.export_metrics(args.export_metrics)


if __name__ == "__main__":
    main()
