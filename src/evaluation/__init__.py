"""
Evaluation modules for model assessment.
"""

from src.evaluation.metrics import (
    TaskMetrics,
    compute_generation_quality,
    compute_classification_metrics,
)

# Import evaluation functions when they're available
try:
    from src.evaluation.evaluate import (
        evaluate_model,
        load_model_and_tokenizer,
    )
except ImportError:
    # Functions not available yet
    evaluate_model = None
    load_model_and_tokenizer = None

__all__ = [
    # Metrics
    "TaskMetrics",
    "compute_generation_quality",
    "compute_classification_metrics",
    
    # Evaluation functions
    "evaluate_model",
    "load_model_and_tokenizer",
]