"""
Prometheus: Multi-Agent Cognitive Emergence System

Phase 1 MVP: Dual-agent collaboration for strategy evolution.
"""

from src.prometheus.agents import (
    AgentMemory,
    CognitiveCell,
    EvaluationSpecialist,
    Message,
    SearchSpecialist,
)

__all__ = [
    "CognitiveCell",
    "SearchSpecialist",
    "EvaluationSpecialist",
    "Message",
    "AgentMemory",
]

__version__ = "0.1.0"
