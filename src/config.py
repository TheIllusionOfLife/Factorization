from dataclasses import dataclass
import os
from dotenv import load_dotenv

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
    return Config(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        max_llm_calls=int(os.getenv("MAX_LLM_CALLS_PER_RUN", "100")),
        llm_enabled=os.getenv("LLM_ENABLED", "true").lower() == "true"
    )
