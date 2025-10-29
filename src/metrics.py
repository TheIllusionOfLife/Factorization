"""Evaluation metrics for strategy assessment."""

from dataclasses import dataclass, field
from typing import Dict, List

# Constants
MAX_SMOOTHNESS_SCORES_TO_STORE = 10
MAX_EXAMPLE_CANDIDATES_TO_STORE = 5


@dataclass
class EvaluationMetrics:
    """Detailed metrics from strategy evaluation."""

    candidate_count: int
    smoothness_scores: List[float] = field(default_factory=list)
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    rejection_stats: Dict[str, int] = field(default_factory=dict)
    example_candidates: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON export."""
        return {
            "candidate_count": self.candidate_count,
            "smoothness_scores": self.smoothness_scores,
            "timing_breakdown": self.timing_breakdown,
            "rejection_stats": self.rejection_stats,
            "example_candidates": self.example_candidates,
        }


__all__ = [
    "EvaluationMetrics",
    "MAX_SMOOTHNESS_SCORES_TO_STORE",
    "MAX_EXAMPLE_CANDIDATES_TO_STORE",
]
