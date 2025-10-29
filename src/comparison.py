"""Comparison engine and baseline strategies."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.crucible import FactorizationCrucible
from src.evolution import EvolutionaryEngine
from src.statistics import ConvergenceDetector, StatisticalAnalyzer
from src.strategy import SMALL_PRIMES, Strategy


class BaselineStrategyGenerator:
    """
    Generates classical GNFS-inspired baseline strategies for comparison.

    These strategies represent "what a human would design" based on
    number theory principles, not evolved solutions.

    Provides three baseline approaches:
    - Conservative: Strict filtering, high smoothness requirements
    - Balanced: Typical quadratic sieve parameters
    - Aggressive: Permissive filtering for high throughput
    """

    def __init__(self):
        self.primes = SMALL_PRIMES

    def get_baseline_strategies(self) -> Dict[str, Strategy]:
        """
        Return dict of named baseline strategies.

        Returns:
            Dict mapping strategy name to Strategy object
        """
        return {
            "conservative": self._conservative_strategy(),
            "balanced": self._balanced_strategy(),
            "aggressive": self._aggressive_strategy(),
        }

    def _conservative_strategy(self) -> Strategy:
        """
        Conservative classical approach.

        Characteristics:
        - Low power (2) for stability
        - Strict modulus filters (eliminate more candidates)
        - High min hits requirement (strict smoothness)

        This strategy casts a narrow net, accepting only candidates
        that are very likely to be smooth.
        """
        return Strategy(
            power=2,
            modulus_filters=[
                (2, [0]),  # Even numbers only
                (3, [0]),  # Divisible by 3
                (5, [0]),  # Divisible by 5
            ],
            smoothness_bound=13,
            min_small_prime_hits=4,  # Strict: need many small factors
        )

    def _balanced_strategy(self) -> Strategy:
        """
        Balanced classical approach (typical quadratic sieve parameters).

        Characteristics:
        - Medium power (3) for balanced candidate generation
        - Moderate modulus filters
        - Reasonable smoothness requirements

        This strategy represents a middle ground between strict and
        permissive filtering.
        """
        return Strategy(
            power=3,
            modulus_filters=[
                (2, [0, 1]),  # Allow both even and odd
                (7, [0, 1, 2, 4]),  # Quadratic residues mod 7
            ],
            smoothness_bound=19,
            min_small_prime_hits=2,
        )

    def _aggressive_strategy(self) -> Strategy:
        """
        Aggressive approach: Cast wide net, accept more candidates.

        Characteristics:
        - High power (4) for diverse candidate generation
        - Minimal modulus filters
        - Low smoothness requirement

        This strategy prioritizes throughput over precision, accepting
        many candidates that may not be smooth.
        """
        return Strategy(
            power=4,
            modulus_filters=[
                (3, [0, 1, 2]),  # Allow all mod 3
            ],
            smoothness_bound=31,
            min_small_prime_hits=1,  # Very permissive
        )


@dataclass
class ComparisonRun:
    """Results from a single comparison run."""

    evolved_fitness: List[float]  # Fitness per generation
    baseline_fitness: Dict[str, float]  # Fitness per baseline strategy
    generations_to_convergence: Optional[int]
    final_best_strategy: Strategy
    random_seed: Optional[int]


class ComparisonEngine:
    """
    Run evolutionary strategies against baseline strategies and compare.

    Supports:
    - Multiple independent runs for statistical rigor
    - Convergence detection for early stopping
    - Statistical significance testing
    """

    def __init__(
        self,
        crucible: FactorizationCrucible,
        num_runs: int = 5,
        max_generations: int = 10,
        population_size: int = 10,
        evaluation_duration: float = 0.1,
        convergence_window: int = 5,
        llm_provider=None,
    ):
        self.crucible = crucible
        self.num_runs = num_runs
        self.max_generations = max_generations
        self.population_size = population_size
        self.evaluation_duration = evaluation_duration
        self.convergence_window = convergence_window
        self.llm_provider = llm_provider

        self.baseline_generator = BaselineStrategyGenerator()
        self.convergence_detector = ConvergenceDetector(window_size=convergence_window)
        self.statistical_analyzer = StatisticalAnalyzer()

    def run_comparison(self, base_seed: Optional[int] = None) -> List[ComparisonRun]:
        """
        Run multiple independent evolutionary runs and compare to baselines.

        Args:
            base_seed: If provided, runs use base_seed+i for reproducibility

        Returns:
            List of ComparisonRun objects (one per run)
        """
        runs = []

        for run_idx in range(self.num_runs):
            seed = base_seed + run_idx if base_seed is not None else None
            print(f"\n{'=' * 60}")
            print(f"COMPARISON RUN {run_idx + 1}/{self.num_runs}")
            if seed is not None:
                print(f"🎲 Random seed: {seed}")
            print(f"{'=' * 60}")

            run_result = self._run_single_comparison(seed)
            runs.append(run_result)

        return runs

    def _run_single_comparison(self, seed: Optional[int]) -> ComparisonRun:
        """Execute one complete evolutionary run vs baselines."""

        # 1. Evaluate baseline strategies
        baseline_fitness = self._evaluate_baselines()

        # 2. Run evolutionary process with convergence detection
        engine = EvolutionaryEngine(
            crucible=self.crucible,
            population_size=self.population_size,
            evaluation_duration=self.evaluation_duration,
            llm_provider=self.llm_provider,
            random_seed=seed,
        )

        engine.initialize_population()
        evolved_fitness_history = []
        converged_at = None

        # Track best strategy seen across all generations
        best_strategy: Optional[Strategy] = None

        for gen in range(self.max_generations):
            # Run evolutionary cycle and get best fitness & strategy from evaluated generation
            best_fitness, best_strategy = engine.run_evolutionary_cycle()
            evolved_fitness_history.append(best_fitness)

            # Check convergence
            if self.convergence_detector.has_converged(evolved_fitness_history):
                converged_at = gen
                print(f"\n✓ Converged at generation {gen}")
                break

        # At least one generation must have run, so best_strategy is not None
        assert best_strategy is not None, "No generations completed"

        return ComparisonRun(
            evolved_fitness=evolved_fitness_history,
            baseline_fitness=baseline_fitness,
            generations_to_convergence=converged_at,
            final_best_strategy=best_strategy,
            random_seed=seed,
        )

    def _evaluate_baselines(self) -> Dict[str, float]:
        """Evaluate all baseline strategies once."""
        baselines = self.baseline_generator.get_baseline_strategies()
        results: Dict[str, float] = {}

        print("\n--- Evaluating Baseline Strategies ---")
        for name, strategy in baselines.items():
            metrics = self.crucible.evaluate_strategy_detailed(
                strategy, self.evaluation_duration
            )
            results[name] = float(metrics.candidate_count)
            print(f"  {name:12s}: fitness = {metrics.candidate_count}")

        return results

    def analyze_results(self, runs: List[ComparisonRun]) -> Dict[str, Any]:
        """
        Perform statistical analysis on comparison runs.

        Returns dict with:
        - comparison_results: Dict[baseline_name, ComparisonResult]
        - convergence_stats: Mean/std generations to convergence
        - num_runs: Number of runs analyzed
        """
        # Extract final evolved fitness from each run
        final_evolved = [run.evolved_fitness[-1] for run in runs]

        # Compare against each baseline
        comparison_results = {}
        for baseline_name in runs[0].baseline_fitness:
            baseline_scores = [run.baseline_fitness[baseline_name] for run in runs]

            result = self.statistical_analyzer.compare_fitness_distributions(
                final_evolved, baseline_scores
            )
            comparison_results[baseline_name] = result

        # Convergence statistics
        convergence_gens = [
            r.generations_to_convergence
            for r in runs
            if r.generations_to_convergence is not None
        ]
        convergence_stats = {
            "mean": float(np.mean(convergence_gens)) if convergence_gens else None,
            "std": float(np.std(convergence_gens)) if convergence_gens else None,
            "convergence_rate": len(convergence_gens) / len(runs),
        }

        return {
            "comparison_results": comparison_results,
            "convergence_stats": convergence_stats,
            "num_runs": len(runs),
        }


def print_llm_summary(llm_provider) -> None:
    """Print LLM cost summary to console."""
    print("\n💰 LLM Cost Summary:")
    print(f"   Total API calls: {llm_provider.call_count}")
    print(
        f"   Total tokens: {llm_provider.input_tokens} in, {llm_provider.output_tokens} out"
    )
    print(f"   Estimated cost: ${llm_provider.total_cost:.6f}")


__all__ = [
    "BaselineStrategyGenerator",
    "ComparisonRun",
    "ComparisonEngine",
    "print_llm_summary",
]
