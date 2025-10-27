from dataclasses import dataclass
import logging
import os
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

def load_config() -> Config:
    load_dotenv()

    # Load and validate max_llm_calls
    try:
        max_llm_calls = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "100"))
    except (ValueError, TypeError):
        logger.warning("Invalid MAX_LLM_CALLS_PER_RUN value, using default 100")
        max_llm_calls = 100

    return Config(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        max_llm_calls=max_llm_calls,
        llm_enabled=os.getenv("LLM_ENABLED", "true").lower() == "true"
    )
