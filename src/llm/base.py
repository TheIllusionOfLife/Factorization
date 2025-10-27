"""Base classes for LLM providers"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from LLM mutation proposal"""

    success: bool
    mutation_params: dict
    reasoning: Optional[str] = None
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def propose_mutation(
        self,
        parent_strategy: dict,
        fitness: int,
        generation: int,
        fitness_history: list,
    ) -> LLMResponse:
        """
        Propose a mutation to improve the given strategy.

        Args:
            parent_strategy: Current strategy parameters
            fitness: Current fitness score
            generation: Current generation number
            fitness_history: List of recent fitness scores

        Returns:
            LLMResponse with mutation parameters or error
        """
        pass

    @property
    @abstractmethod
    def total_cost(self) -> float:
        """Total cost of all API calls"""
        pass

    @property
    @abstractmethod
    def call_count(self) -> int:
        """Number of API calls made"""
        pass
