"""
Data structures for meta-learning operator selection.

This module provides dataclasses for tracking which reproduction operators
(crossover, mutation, random) create strategies and their performance.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class OperatorMetadata:
    """Metadata about how a strategy was created.

    Attributes:
        operator: Which operator created this strategy ("crossover", "mutation", "random")
        parent_ids: List of parent civilization IDs
        parent_fitness: Fitness scores of parent strategies
        generation: Generation number when strategy was created
    """

    operator: str  # "crossover", "mutation", or "random"
    parent_ids: List[str]
    parent_fitness: List[float]
    generation: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "operator": self.operator,
            "parent_ids": self.parent_ids,
            "parent_fitness": self.parent_fitness,
            "generation": self.generation,
        }


@dataclass
class OperatorStatistics:
    """Statistics tracking operator performance.

    Attributes:
        total_offspring: Total number of offspring created by this operator
        elite_offspring: Number of offspring that became elite (top 20%)
        total_fitness_improvement: Sum of fitness improvements over parents
        avg_fitness_improvement: Average fitness improvement
        success_rate: Fraction of offspring that became elite
    """

    total_offspring: int = 0
    elite_offspring: int = 0
    total_fitness_improvement: float = 0.0
    avg_fitness_improvement: float = 0.0
    success_rate: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_offspring": self.total_offspring,
            "elite_offspring": self.elite_offspring,
            "total_fitness_improvement": self.total_fitness_improvement,
            "avg_fitness_improvement": self.avg_fitness_improvement,
            "success_rate": self.success_rate,
        }


@dataclass
class AdaptiveRates:
    """Dynamic operator selection rates based on performance.

    Attributes:
        crossover_rate: Probability of selecting crossover operator
        mutation_rate: Probability of selecting mutation operator
        random_rate: Probability of selecting random operator
        generation: Generation when these rates were calculated
        operator_stats: Statistics for each operator at this generation
    """

    crossover_rate: float
    mutation_rate: float
    random_rate: float
    generation: int
    operator_stats: Dict[str, OperatorStatistics]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "random_rate": self.random_rate,
            "generation": self.generation,
            "operator_stats": {
                op: stats.to_dict() for op, stats in self.operator_stats.items()
            },
        }
