import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Centralized configuration for evolutionary GNFS optimizer.

    This dataclass holds ALL tunable parameters for the system,
    providing a single source of truth for configuration.

    All parameters are validated in __post_init__() and will raise
    ValueError if constraints are violated.
    """

    # Epsilon tolerance for floating point comparisons (1% tolerance)
    # Accounts for edge cases like 3 * 0.33 = 0.99 (should pass 1.0 check)
    EPSILON: ClassVar[float] = 0.01

    # ===== LLM Configuration (existing) =====
    api_key: str
    max_llm_calls: int = 100
    llm_enabled: bool = True
    temperature_base: float = 0.8
    temperature_max: float = 1.2
    max_tokens: int = 1024
    temperature_scaling_generations: int = 10

    # ===== Evolution Parameters (NEW) =====
    elite_selection_rate: float = 0.2  # Top X% become parents
    crossover_rate: float = 0.3  # Fraction from two parents
    mutation_rate: float = 0.5  # Fraction from one parent
    evaluation_duration: float = 0.1  # Seconds per evaluation

    # ===== Meta-Learning Parameters (NEW) =====
    meta_learning_min_rate: float = 0.1  # Minimum operator rate
    meta_learning_max_rate: float = 0.7  # Maximum operator rate
    adaptation_window: int = 5  # Generations for rate adaptation
    fallback_inf_rate: float = 0.8  # When infinite scores exist
    fallback_finite_rate: float = 0.2  # When infinite scores exist

    # ===== Strategy Bounds (NEW) =====
    power_min: int = 2  # Minimum polynomial power
    power_max: int = 5  # Maximum polynomial power
    max_filters: int = 4  # Maximum modulus filters
    min_hits_min: int = 1  # Minimum required prime hits
    min_hits_max: int = 6  # Maximum required prime hits

    # ===== Mutation Probabilities (NEW) =====
    mutation_prob_power: float = 0.3  # Probability of mutating power
    mutation_prob_filter: float = 0.3  # Probability of mutating filters
    mutation_prob_modulus: float = 0.5  # Probability of changing modulus
    mutation_prob_residue: float = 0.5  # Probability of changing residues
    mutation_prob_add_filter: float = 0.15  # Probability of adding filter

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_evolution_params()
        self._validate_meta_learning_params()
        self._validate_strategy_bounds()
        self._validate_mutation_probs()

    def _validate_evolution_params(self):
        """Validate evolution parameters."""
        if not 0.0 < self.elite_selection_rate <= 1.0:
            raise ValueError(
                f"elite_selection_rate must be in (0, 1], got {self.elite_selection_rate}"
            )

        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(
                f"crossover_rate must be in [0, 1], got {self.crossover_rate}"
            )

        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(
                f"mutation_rate must be in [0, 1], got {self.mutation_rate}"
            )

        if self.crossover_rate + self.mutation_rate > 1.0 + self.EPSILON:
            raise ValueError(
                f"Sum of crossover_rate + mutation_rate must be <= 1.0, "
                f"got {self.crossover_rate} + {self.mutation_rate} = "
                f"{self.crossover_rate + self.mutation_rate}"
            )

        if self.evaluation_duration <= 0:
            raise ValueError(
                f"evaluation_duration must be > 0, got {self.evaluation_duration}"
            )

    def _validate_meta_learning_params(self):
        """Validate meta-learning parameters."""
        if not (
            0.0 <= self.meta_learning_min_rate <= self.meta_learning_max_rate <= 1.0
        ):
            raise ValueError(
                f"Rate bounds must satisfy 0 ≤ min ≤ max ≤ 1, "
                f"got min={self.meta_learning_min_rate}, max={self.meta_learning_max_rate}"
            )

        if 3 * self.meta_learning_min_rate > 1.0 + self.EPSILON:
            raise ValueError(
                f"Infeasible bounds: 3 * min_rate > 1.0, "
                f"got 3 * {self.meta_learning_min_rate} = {3 * self.meta_learning_min_rate}"
            )

        if 3 * self.meta_learning_max_rate < 1.0 - self.EPSILON:
            raise ValueError(
                f"Infeasible bounds: 3 * max_rate < 1.0, "
                f"got 3 * {self.meta_learning_max_rate} = {3 * self.meta_learning_max_rate}"
            )

        if self.adaptation_window < 1:
            raise ValueError(
                f"adaptation_window must be >= 1, got {self.adaptation_window}"
            )

    def _validate_strategy_bounds(self):
        """Validate strategy bounds."""
        if not 2 <= self.power_min <= self.power_max <= 5:
            raise ValueError(
                f"Power bounds must satisfy 2 ≤ min ≤ max ≤ 5, "
                f"got min={self.power_min}, max={self.power_max}"
            )

        if self.max_filters < 1:
            raise ValueError(f"max_filters must be >= 1, got {self.max_filters}")

        if not 1 <= self.min_hits_min <= self.min_hits_max:
            raise ValueError(
                f"Min hits bounds must satisfy 1 ≤ min ≤ max, "
                f"got min={self.min_hits_min}, max={self.min_hits_max}"
            )

    def _validate_mutation_probs(self):
        """Validate mutation probabilities."""
        probs = [
            ("mutation_prob_power", self.mutation_prob_power),
            ("mutation_prob_filter", self.mutation_prob_filter),
            ("mutation_prob_modulus", self.mutation_prob_modulus),
            ("mutation_prob_residue", self.mutation_prob_residue),
            ("mutation_prob_add_filter", self.mutation_prob_add_filter),
        ]
        for name, prob in probs:
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {prob}")

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert config to dictionary.

        Args:
            include_sensitive: If True, include api_key. Default False for export.

        Returns:
            Dictionary representation of config (excludes EPSILON class variable).
        """
        config_dict = asdict(self)

        # Exclude class variables (EPSILON) - ClassVar included in asdict() on Python 3.9-3.10
        config_dict.pop("EPSILON", None)

        if not include_sensitive:
            config_dict.pop("api_key", None)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], api_key: str) -> "Config":
        """Create Config from dictionary.

        Args:
            config_dict: Dictionary of config values
            api_key: API key (not stored in dict for security)

        Returns:
            Config instance
        """
        return cls(api_key=api_key, **config_dict)


def load_config() -> Config:
    load_dotenv()

    # Load and validate max_llm_calls
    try:
        max_llm_calls = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "100"))
    except (ValueError, TypeError):
        logger.warning("Invalid MAX_LLM_CALLS_PER_RUN value, using default 100")
        max_llm_calls = 100

    # Load LLM settings
    api_key = os.getenv("GEMINI_API_KEY", "")
    llm_enabled = os.getenv("LLM_ENABLED", "true").lower() == "true"

    # Validate API key if LLM is enabled
    if llm_enabled and not api_key:
        raise ValueError(
            "LLM is enabled but GEMINI_API_KEY is not set. "
            "Either set GEMINI_API_KEY in .env or disable LLM with LLM_ENABLED=false"
        )

    return Config(api_key=api_key, max_llm_calls=max_llm_calls, llm_enabled=llm_enabled)
