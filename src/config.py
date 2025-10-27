import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    api_key: str
    max_llm_calls: int = 100
    llm_enabled: bool = True
    temperature_base: float = 0.8
    temperature_max: float = 1.2
    max_tokens: int = 1024
    temperature_scaling_generations: int = (
        10  # Number of generations to scale temperature from base to max
    )


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
