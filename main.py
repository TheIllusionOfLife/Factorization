"""Main CLI entry point for evolutionary GNFS strategy optimizer."""

import argparse
import json
import os
import sys
from pathlib import Path

from src.config import Config
from src.crucible import FactorizationCrucible
from src.evolution import EvolutionaryEngine
from src.logging_config import setup_logging


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
        metavar="SECS",
        help="Evaluation duration in seconds (default: 0.1, overrides config if provided)",
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

    # Evolution parameters
    parser.add_argument(
        "--elite-rate",
        type=float,
        metavar="RATE",
        help="Elite selection rate: fraction of top performers that become parents (default: 0.2)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        metavar="RATE",
        help="Crossover rate: fraction of offspring from two parents (default: 0.3)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        metavar="RATE",
        help="Mutation rate: fraction of offspring from single parent (default: 0.5)",
    )

    # Strategy bounds
    parser.add_argument(
        "--power-min",
        type=int,
        metavar="N",
        help="Minimum polynomial power (default: 2, range: 2-5)",
    )
    parser.add_argument(
        "--power-max",
        type=int,
        metavar="N",
        help="Maximum polynomial power (default: 5, range: 2-5)",
    )
    parser.add_argument(
        "--max-filters",
        type=int,
        metavar="N",
        help="Maximum number of modulus filters per strategy (default: 4)",
    )
    parser.add_argument(
        "--min-hits-min",
        type=int,
        metavar="N",
        help="Minimum required small prime hits (default: 1)",
    )
    parser.add_argument(
        "--min-hits-max",
        type=int,
        metavar="N",
        help="Maximum required small prime hits (default: 6)",
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
        metavar="N",
        help="Generations to consider for rate adaptation (default: 5)",
    )
    parser.add_argument(
        "--meta-min-rate",
        type=float,
        metavar="RATE",
        help="Meta-learning minimum operator rate (default: 0.1)",
    )
    parser.add_argument(
        "--meta-max-rate",
        type=float,
        metavar="RATE",
        help="Meta-learning maximum operator rate (default: 0.7)",
    )
    parser.add_argument(
        "--fallback-inf-rate",
        type=float,
        metavar="RATE",
        help="Rate split for untried operators in meta-learning (default: 0.8)",
    )
    parser.add_argument(
        "--fallback-finite-rate",
        type=float,
        metavar="RATE",
        help="Rate split for tried operators in meta-learning (default: 0.2)",
    )

    # Mutation probability arguments
    parser.add_argument(
        "--mutation-prob-power",
        type=float,
        metavar="PROB",
        help="Probability of mutating power parameter (default: 0.3)",
    )
    parser.add_argument(
        "--mutation-prob-filter",
        type=float,
        metavar="PROB",
        help="Probability of mutating filter parameter (default: 0.3)",
    )
    parser.add_argument(
        "--mutation-prob-modulus",
        type=float,
        metavar="PROB",
        help="Probability of changing modulus in filter mutation (default: 0.5)",
    )
    parser.add_argument(
        "--mutation-prob-residue",
        type=float,
        metavar="PROB",
        help="Probability of changing residues in filter mutation (default: 0.5)",
    )
    parser.add_argument(
        "--mutation-prob-add-filter",
        type=float,
        metavar="PROB",
        help="Probability of adding a new filter during mutation (default: 0.15)",
    )

    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Set logging level (default: from LOG_LEVEL env var or INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Write logs to file (default: from LOG_FILE env var or no file logging)",
    )

    # Prometheus (multi-agent) arguments
    parser.add_argument(
        "--prometheus",
        action="store_true",
        help="Enable Prometheus multi-agent mode (dual-agent collaboration)",
    )
    parser.add_argument(
        "--prometheus-mode",
        type=str,
        choices=["collaborative", "search_only", "eval_only", "rulebased"],
        default="collaborative",
        metavar="MODE",
        help="Prometheus mode: collaborative (default), search_only, eval_only, rulebased",
    )
    parser.add_argument(
        "--max-api-cost",
        type=float,
        metavar="DOLLARS",
        help="Maximum API cost in dollars (safety limit, default: 1.0)",
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

    # Initialize logging BEFORE any other operations
    log_level = args.log_level or os.getenv("LOG_LEVEL", "INFO")
    log_file = args.log_file or os.getenv("LOG_FILE", None)

    # Convert empty string from .env to None
    if log_file == "":
        log_file = None

    setup_logging(level=log_level, log_file=log_file, console_output=True)

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

    # Build config from environment and CLI overrides
    try:
        # Create config from CLI args and environment (immutable construction pattern)
        # Validation happens once in Config.__post_init__() - no mutation after construction
        config = Config.from_args_and_env(args, use_llm=args.llm)

    except ValueError as e:
        print(f"❌ ERROR: Invalid configuration: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ ERROR: Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Initialize LLM provider if requested
    llm_provider = None
    if args.llm:
        try:
            from src.comparison import print_llm_summary
            from src.llm.gemini import GeminiProvider

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

    # Prometheus multi-agent mode
    if args.prometheus:
        from src.prometheus.experiment import PrometheusExperiment

        print("\n🔬 Prometheus Multi-Agent Mode")
        print(f"🎯 Target number: {args.number}")
        print(f"🧬 Generations: {args.generations}, Population: {args.population}")
        print(f"⏱️  Evaluation duration: {config.evaluation_duration}s per strategy")
        print(f"🤖 Mode: {config.prometheus_mode}")
        if args.llm:
            print(f"💰 Max API cost: ${config.max_api_cost:.2f}")
        if args.seed is not None:
            print(f"🎲 Random seed: {args.seed} (reproducible run)")
        print()

        # Create experiment
        experiment = PrometheusExperiment(
            config=config,
            target_number=args.number,
            random_seed=args.seed,
        )

        # Run experiment based on mode
        if config.prometheus_mode == "collaborative":
            # Collaborative mode: dual-agent evolution
            print("🔄 Running collaborative dual-agent evolution...")
            best_fitness, best_strategy, comm_stats = (
                experiment.run_collaborative_evolution(
                    generations=args.generations,
                    population_size=args.population,
                )
            )

            print("\n✅ Evolution complete!")
            print(f"📊 Best fitness: {best_fitness}")
            print(f"📈 Total messages exchanged: {comm_stats['total_messages']}")
            print(f"💬 Messages by type: {comm_stats['messages_by_type']}")
            print("\n🏆 Best strategy:")
            print(f"   Power: {best_strategy.power}")
            print(f"   Modulus filters: {best_strategy.modulus_filters}")
            print(f"   Smoothness bound: {best_strategy.smoothness_bound}")
            print(f"   Min small prime hits: {best_strategy.min_small_prime_hits}")

            # Export metrics if requested
            if args.export_metrics:
                print(f"\n📁 Exporting metrics to {args.export_metrics}...")
                export_data = {
                    "mode": "prometheus_collaborative",
                    "target_number": args.number,
                    "generations": args.generations,
                    "population_size": args.population,
                    "random_seed": args.seed,
                    "config": config.to_dict(include_sensitive=False),
                    "best_fitness": best_fitness,
                    "best_strategy": {
                        "power": best_strategy.power,
                        "modulus_filters": best_strategy.modulus_filters,
                        "smoothness_bound": best_strategy.smoothness_bound,
                        "min_small_prime_hits": best_strategy.min_small_prime_hits,
                    },
                    "communication_stats": comm_stats,
                }
                Path(args.export_metrics).parent.mkdir(parents=True, exist_ok=True)
                with open(args.export_metrics, "w") as f:
                    json.dump(export_data, f, indent=2)
                print("✅ Metrics exported successfully")

        else:
            # Baseline mode: single-agent or rule-based
            print(f"🔄 Running {config.prometheus_mode} baseline...")
            best_fitness, best_strategy = experiment.run_independent_baseline(
                agent_type=config.prometheus_mode,
                generations=args.generations,
                population_size=args.population,
            )

            print("\n✅ Evolution complete!")
            print(f"📊 Best fitness: {best_fitness}")
            print("\n🏆 Best strategy:")
            print(f"   Power: {best_strategy.power}")
            print(f"   Modulus filters: {best_strategy.modulus_filters}")
            print(f"   Smoothness bound: {best_strategy.smoothness_bound}")
            print(f"   Min small prime hits: {best_strategy.min_small_prime_hits}")

            # Export metrics if requested
            if args.export_metrics:
                print(f"\n📁 Exporting metrics to {args.export_metrics}...")
                export_data = {
                    "mode": f"prometheus_{config.prometheus_mode}",
                    "target_number": args.number,
                    "generations": args.generations,
                    "population_size": args.population,
                    "random_seed": args.seed,
                    "config": config.to_dict(include_sensitive=False),
                    "best_fitness": best_fitness,
                    "best_strategy": {
                        "power": best_strategy.power,
                        "modulus_filters": best_strategy.modulus_filters,
                        "smoothness_bound": best_strategy.smoothness_bound,
                        "min_small_prime_hits": best_strategy.min_small_prime_hits,
                    },
                }
                Path(args.export_metrics).parent.mkdir(parents=True, exist_ok=True)
                with open(args.export_metrics, "w") as f:
                    json.dump(export_data, f, indent=2)
                print("✅ Metrics exported successfully")

        # Display LLM cost summary if used
        if llm_provider:
            from src.comparison import print_llm_summary

            print()
            print_llm_summary(llm_provider)

        sys.exit(0)

    # Comparison mode vs normal evolution mode
    if args.compare_baseline:
        # Import optional comparison dependencies only when needed
        from src.comparison import ComparisonEngine

        # Comparison mode: Run vs baselines with statistical analysis
        print(f"\n🎯 Target number: {args.number}")
        print(f"🧬 Generations: {args.generations}, Population: {args.population}")
        print(f"⏱️  Evaluation duration: {config.evaluation_duration}s per strategy")
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
            evaluation_duration=config.evaluation_duration,
            convergence_window=args.convergence_window,
            llm_provider=llm_provider,
            config=config,
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
            config=config,
            llm_provider=llm_provider,
            random_seed=args.seed,
            enable_meta_learning=args.meta_learning,
        )

        print(f"\n🎯 Target number: {args.number}")
        print(f"🧬 Generations: {args.generations}, Population: {args.population}")
        print(f"⏱️  Evaluation duration: {config.evaluation_duration}s per strategy")
        if args.meta_learning:
            print(
                f"🔀 Reproduction: {config.crossover_rate:.0%} crossover (initial), {config.mutation_rate:.0%} mutation (initial), {engine.random_rate:.0%} random (initial)"
            )
            print(
                f"🧠 Meta-learning enabled: Rates will adapt based on performance (window={config.adaptation_window})"
            )
        else:
            print(
                f"🔀 Reproduction: {config.crossover_rate:.0%} crossover, {config.mutation_rate:.0%} mutation, {engine.random_rate:.0%} random"
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
