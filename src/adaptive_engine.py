"""
Meta-learning engine for adaptive operator selection.

This module implements the MetaLearningEngine which tracks operator performance
and adapts selection rates using the UCB1 (Upper Confidence Bound) algorithm.
"""

import math
from typing import Dict, List

from src.meta_learning import AdaptiveRates, OperatorStatistics


class MetaLearningEngine:
    """Tracks operator performance and adapts selection rates.

    Uses UCB1 (Upper Confidence Bound) algorithm to balance exploration
    (trying all operators) with exploitation (favoring successful ones).

    Attributes:
        adaptation_window: Number of generations to analyze for rate adaptation
        min_rate: Minimum allowed rate for any operator (prevents ignoring operators)
        max_rate: Maximum allowed rate for any operator (prevents over-exploitation)
        current_generation: Current generation number
        operator_history: Historical statistics per generation
        current_stats: Statistics being accumulated for current generation
    """

    def __init__(
        self,
        adaptation_window: int = 5,
        min_rate: float = 0.1,
        max_rate: float = 0.7,
    ):
        """Initialize meta-learning engine.

        Args:
            adaptation_window: Generations to consider for adaptation (default: 5)
            min_rate: Minimum rate for any operator (default: 0.1)
            max_rate: Maximum rate for any operator (default: 0.7)

        Raises:
            ValueError: If rate bounds are invalid or infeasible
        """
        # Validate rate bounds
        if not (0.0 <= min_rate <= max_rate <= 1.0):
            raise ValueError(
                f"Rate bounds must satisfy 0 ≤ min_rate ≤ max_rate ≤ 1, "
                f"got min_rate={min_rate}, max_rate={max_rate}"
            )

        # Check feasibility: 3 operators must fit within constraints
        if 3 * min_rate > 1.0 + 1e-9:
            raise ValueError(
                f"Infeasible bounds: 3 * min_rate ({3 * min_rate:.3f}) > 1.0. "
                f"Cannot satisfy min_rate={min_rate} for all 3 operators."
            )
        if 3 * max_rate < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible bounds: 3 * max_rate ({3 * max_rate:.3f}) < 1.0. "
                f"Cannot distribute rates with max_rate={max_rate} for all 3 operators."
            )

        self.adaptation_window = adaptation_window
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_generation = 0
        self.operator_history: List[Dict[str, OperatorStatistics]] = []

        # Current generation statistics (accumulated during generation)
        self.current_stats: Dict[str, OperatorStatistics] = {
            "crossover": OperatorStatistics(),
            "mutation": OperatorStatistics(),
            "random": OperatorStatistics(),
        }

    def update_statistics(
        self, operator: str, fitness_improvement: float, became_elite: bool
    ) -> None:
        """Update operator statistics with offspring performance.

        Args:
            operator: Operator that created offspring ("crossover", "mutation", "random")
            fitness_improvement: Fitness difference from parent (can be negative)
            became_elite: Whether offspring became elite (top 20%)
        """
        stats = self.current_stats[operator]
        stats.total_offspring += 1
        if became_elite:
            stats.elite_offspring += 1
        stats.total_fitness_improvement += fitness_improvement

        # Update average and success rate
        stats.avg_fitness_improvement = (
            stats.total_fitness_improvement / stats.total_offspring
        )
        stats.success_rate = stats.elite_offspring / stats.total_offspring

    def finalize_generation(self) -> None:
        """Finalize current generation and prepare for next.

        Saves current statistics to history and resets accumulators.
        """
        # Save current stats to history
        self.operator_history.append(self.current_stats.copy())

        # Reset for next generation
        self.current_generation += 1
        self.current_stats = {
            "crossover": OperatorStatistics(),
            "mutation": OperatorStatistics(),
            "random": OperatorStatistics(),
        }

    def get_current_statistics(self) -> Dict[str, OperatorStatistics]:
        """Get current generation statistics.

        Returns:
            Dictionary mapping operator names to their statistics
        """
        return self.current_stats

    def get_operator_history(self) -> List[Dict[str, OperatorStatistics]]:
        """Get historical operator statistics.

        Returns:
            List of statistics dictionaries, one per generation
        """
        return self.operator_history

    def calculate_adaptive_rates(
        self, current_rates: Dict[str, float]
    ) -> AdaptiveRates:
        """Calculate adaptive rates using UCB1 algorithm.

        Args:
            current_rates: Current operator rates

        Returns:
            AdaptiveRates with new rates and statistics
        """
        # If not enough history, return current rates
        if len(self.operator_history) < self.adaptation_window:
            return AdaptiveRates(
                crossover_rate=current_rates["crossover"],
                mutation_rate=current_rates["mutation"],
                random_rate=current_rates["random"],
                generation=self.current_generation,
                operator_stats=self.current_stats.copy(),
            )

        # Aggregate statistics over adaptation window
        window_stats = self._aggregate_window_statistics()

        # Calculate UCB1 scores
        ucb_scores = self._calculate_ucb_scores(window_stats)

        # Convert scores to rates
        new_rates = self._scores_to_rates(ucb_scores)

        return AdaptiveRates(
            crossover_rate=new_rates["crossover"],
            mutation_rate=new_rates["mutation"],
            random_rate=new_rates["random"],
            generation=self.current_generation,
            operator_stats=window_stats,
        )

    def _aggregate_window_statistics(self) -> Dict[str, OperatorStatistics]:
        """Aggregate statistics over the adaptation window.

        Returns:
            Aggregated statistics for each operator
        """
        # Get recent generations (last N)
        recent_history = self.operator_history[-self.adaptation_window :]

        aggregated = {
            "crossover": OperatorStatistics(),
            "mutation": OperatorStatistics(),
            "random": OperatorStatistics(),
        }

        for gen_stats in recent_history:
            for operator in ["crossover", "mutation", "random"]:
                stats = gen_stats[operator]
                agg = aggregated[operator]

                agg.total_offspring += stats.total_offspring
                agg.elite_offspring += stats.elite_offspring
                agg.total_fitness_improvement += stats.total_fitness_improvement

        # Calculate averages and success rates
        for _operator, stats in aggregated.items():
            if stats.total_offspring > 0:
                stats.avg_fitness_improvement = (
                    stats.total_fitness_improvement / stats.total_offspring
                )
                stats.success_rate = stats.elite_offspring / stats.total_offspring
            else:
                stats.avg_fitness_improvement = 0.0
                stats.success_rate = 0.0

        return aggregated

    def _calculate_ucb_scores(
        self, window_stats: Dict[str, OperatorStatistics]
    ) -> Dict[str, float]:
        """Calculate UCB1 scores for each operator.

        UCB1 formula: score = success_rate + sqrt(2 * ln(total_trials) / operator_trials)

        The exploration bonus (sqrt term) decreases as operator is tried more,
        favoring well-tried operators while still exploring less-tried ones.

        Args:
            window_stats: Aggregated statistics over window

        Returns:
            UCB1 scores for each operator
        """
        # Total trials across all operators
        total_trials = sum(stats.total_offspring for stats in window_stats.values())

        # Avoid division by zero
        if total_trials == 0:
            return {"crossover": 1.0, "mutation": 1.0, "random": 1.0}

        ucb_scores = {}
        for operator, stats in window_stats.items():
            if stats.total_offspring == 0:
                # Untried operator gets infinite score (explore it!)
                ucb_scores[operator] = float("inf")
            else:
                # UCB1 formula
                exploitation = stats.success_rate
                exploration = math.sqrt(
                    2 * math.log(total_trials) / stats.total_offspring
                )
                ucb_scores[operator] = exploitation + exploration

        return ucb_scores

    def _scores_to_rates(self, ucb_scores: Dict[str, float]) -> Dict[str, float]:
        """Convert UCB1 scores to selection rates.

        Uses softmax-like normalization to convert scores to probabilities,
        then enforces min/max rate constraints.

        Args:
            ucb_scores: UCB1 scores for each operator

        Returns:
            Normalized selection rates that sum to 1.0
        """
        # Handle infinite scores (untried operators)
        has_inf = any(math.isinf(score) for score in ucb_scores.values())
        if has_inf:
            inf_count = sum(1 for score in ucb_scores.values() if math.isinf(score))
            if inf_count == 3:
                # All operators untried - use uniform distribution
                return {"crossover": 1 / 3, "mutation": 1 / 3, "random": 1 / 3}

            # Some operators untried - give them priority (80% total)
            finite_count = 3 - inf_count
            inf_rate = 0.8 / inf_count
            finite_rate = 0.2 / finite_count

            rates = {
                operator: inf_rate if math.isinf(score) else finite_rate
                for operator, score in ucb_scores.items()
            }
            # Enforce bounds and renormalize
            return self._enforce_rate_bounds(rates)

        # Normalize scores to [0, 1] range
        min_score = min(ucb_scores.values())
        max_score = max(ucb_scores.values())
        score_range = max_score - min_score

        if score_range == 0:
            # All scores equal - use uniform distribution
            return {"crossover": 1 / 3, "mutation": 1 / 3, "random": 1 / 3}

        normalized = {
            op: (score - min_score) / score_range for op, score in ucb_scores.items()
        }

        # Apply softmax to convert to probabilities
        exp_scores = {op: math.exp(norm) for op, norm in normalized.items()}
        total_exp = sum(exp_scores.values())
        rates = {op: exp_val / total_exp for op, exp_val in exp_scores.items()}

        # Enforce min/max constraints
        rates = self._enforce_rate_bounds(rates)

        return rates

    def _enforce_rate_bounds(self, rates: Dict[str, float]) -> Dict[str, float]:
        """Enforce min/max rate constraints.

        Ensures all rates are within [min_rate, max_rate] and sum to 1.0.
        Uses iterative adjustment to satisfy both constraints simultaneously.

        Args:
            rates: Unconstrained rates

        Returns:
            Constrained rates that sum to 1.0
        """
        operators = list(rates.keys())
        result = {
            op: max(self.min_rate, min(self.max_rate, rates[op])) for op in operators
        }

        # Iteratively adjust to sum to 1.0 while respecting bounds
        for _ in range(20):  # Max iterations to prevent infinite loop
            total = sum(result.values())
            if abs(total - 1.0) < 1e-9:
                break

            # Find operators that can be adjusted (not at bounds)
            adjustable = [
                op
                for op in operators
                if (total > 1.0 and result[op] > self.min_rate)
                or (total < 1.0 and result[op] < self.max_rate)
            ]

            if not adjustable:
                # Can't adjust further - normalize what we have
                # This handles infeasible constraint cases
                result = {op: v / total for op, v in result.items()}
                break

            # Distribute surplus/deficit equally among adjustable operators
            adjustment = (1.0 - total) / len(adjustable)
            for op in adjustable:
                new_val = result[op] + adjustment
                # Clip to bounds
                result[op] = max(self.min_rate, min(self.max_rate, new_val))

        return result
