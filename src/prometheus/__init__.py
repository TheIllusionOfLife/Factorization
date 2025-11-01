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
from src.prometheus.communication import SimpleCommunicationChannel
from src.prometheus.experiment import EmergenceMetrics, PrometheusExperiment

__all__ = [
    "CognitiveCell",
    "SearchSpecialist",
    "EvaluationSpecialist",
    "Message",
    "AgentMemory",
    "SimpleCommunicationChannel",
    "PrometheusExperiment",
    "EmergenceMetrics",
]

__version__ = "0.1.0"
