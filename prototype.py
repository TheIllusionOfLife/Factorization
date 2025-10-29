"""
Compatibility shim for prototype.py.

This file maintains backward compatibility for imports while the codebase
has been refactored into modular components under src/.

All functionality has been moved to:
- src/metrics.py: EvaluationMetrics
- src/strategy.py: Strategy, StrategyGenerator, LLMStrategyGenerator, crossover
- src/crucible.py: FactorizationCrucible
- src/evolution.py: EvolutionaryEngine
- src/comparison.py: ComparisonEngine, BaselineStrategyGenerator, ComparisonRun
- main.py: CLI entry point

For new code, import directly from src/ modules instead of prototype.py.
"""

# Re-export all components for backward compatibility
from src.comparison import (
    BaselineStrategyGenerator,
    ComparisonEngine,
    ComparisonRun,
    print_llm_summary,
)
from src.crucible import FactorizationCrucible
from src.evolution import EvolutionaryEngine
from src.metrics import (
    MAX_EXAMPLE_CANDIDATES_TO_STORE,
    MAX_SMOOTHNESS_SCORES_TO_STORE,
    EvaluationMetrics,
)
from src.strategy import (
    SMALL_PRIMES,
    LLMStrategyGenerator,
    Strategy,
    StrategyGenerator,
    blend_modulus_filters,
    crossover_strategies,
)

__all__ = [
    # Metrics
    "EvaluationMetrics",
    "MAX_SMOOTHNESS_SCORES_TO_STORE",
    "MAX_EXAMPLE_CANDIDATES_TO_STORE",
    # Strategy
    "SMALL_PRIMES",
    "Strategy",
    "blend_modulus_filters",
    "crossover_strategies",
    "StrategyGenerator",
    "LLMStrategyGenerator",
    # Crucible
    "FactorizationCrucible",
    # Evolution
    "EvolutionaryEngine",
    # Comparison
    "BaselineStrategyGenerator",
    "ComparisonRun",
    "ComparisonEngine",
    "print_llm_summary",
]

# CLI entry point for backward compatibility
if __name__ == "__main__":
    from main import main

    main()
