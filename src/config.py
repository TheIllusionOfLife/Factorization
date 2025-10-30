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

        # Validate fallback split for untried vs tried operators
        if not (0.0 <= self.fallback_inf_rate <= 1.0) or not (
            0.0 <= self.fallback_finite_rate <= 1.0
        ):
            raise ValueError(
                f"fallback rates must be in [0,1], "
                f"got inf={self.fallback_inf_rate}, finite={self.fallback_finite_rate}"
            )

        # Use tight epsilon (1e-9) for exact equality - these must sum to exactly 1.0
        if abs(self.fallback_inf_rate + self.fallback_finite_rate - 1.0) >= 1e-9:
            raise ValueError(
                f"fallback_inf_rate + fallback_finite_rate must equal 1.0, "
                f"got {self.fallback_inf_rate + self.fallback_finite_rate:.6f}"
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

    @classmethod
    def from_args_and_env(cls, args, use_llm: bool) -> "Config":
        """Create Config from CLI args and environment variables.

        This is the recommended way to create Config with CLI overrides,
        avoiding the anti-pattern of mutating config after construction.

        Args:
            args: Argparse Namespace with CLI arguments
            use_llm: Whether to load LLM config from environment

        Returns:
            Config instance with validation applied once at construction

        Example:
            config = Config.from_args_and_env(args, args.llm)
        """
        # Start with environment config if using LLM
        if use_llm:
            base_dict = load_config().to_dict(include_sensitive=True)
            api_key = base_dict.pop("api_key")
        else:
            base_dict = {}
            api_key = ""

        # Build overrides dict from CLI args (only non-None values)
        overrides = {}
        if args.duration is not None:
            overrides["evaluation_duration"] = args.duration
        if args.elite_rate is not None:
            overrides["elite_selection_rate"] = args.elite_rate
        if args.crossover_rate is not None:
            overrides["crossover_rate"] = args.crossover_rate
        if args.mutation_rate is not None:
            overrides["mutation_rate"] = args.mutation_rate
        if args.power_min is not None:
            overrides["power_min"] = args.power_min
        if args.power_max is not None:
            overrides["power_max"] = args.power_max
        if args.max_filters is not None:
            overrides["max_filters"] = args.max_filters
        if args.min_hits_min is not None:
            overrides["min_hits_min"] = args.min_hits_min
        if args.min_hits_max is not None:
            overrides["min_hits_max"] = args.min_hits_max
        if args.adaptation_window is not None:
            overrides["adaptation_window"] = args.adaptation_window
        if args.meta_min_rate is not None:
            overrides["meta_learning_min_rate"] = args.meta_min_rate
        if args.meta_max_rate is not None:
            overrides["meta_learning_max_rate"] = args.meta_max_rate
        if args.fallback_inf_rate is not None:
            overrides["fallback_inf_rate"] = args.fallback_inf_rate
        if args.fallback_finite_rate is not None:
            overrides["fallback_finite_rate"] = args.fallback_finite_rate
        if args.mutation_prob_power is not None:
            overrides["mutation_prob_power"] = args.mutation_prob_power
        if args.mutation_prob_filter is not None:
            overrides["mutation_prob_filter"] = args.mutation_prob_filter
        if args.mutation_prob_modulus is not None:
            overrides["mutation_prob_modulus"] = args.mutation_prob_modulus
        if args.mutation_prob_residue is not None:
            overrides["mutation_prob_residue"] = args.mutation_prob_residue
        if args.mutation_prob_add_filter is not None:
            overrides["mutation_prob_add_filter"] = args.mutation_prob_add_filter

        # Merge base + overrides and construct once (validation happens in __post_init__)
        return cls(api_key=api_key, llm_enabled=use_llm, **{**base_dict, **overrides})


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
