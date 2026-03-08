"""
Evaluation modules for model assessment.
"""

from src.evaluation.metrics import (
    TaskMetrics,
    compute_generation_quality,
    compute_classification_metrics,
)

__all__ = [
    "TaskMetrics",
    "compute_generation_quality",
    "compute_classification_metrics",
]