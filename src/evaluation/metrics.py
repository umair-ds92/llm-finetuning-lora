"""
Evaluation metrics for model performance assessment.
"""

import re
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TaskMetrics:
    """Compute task-specific evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = {}
    
    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match accuracy.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Exact match score (0-1)
        """
        matches = sum(
            pred.strip().lower() == ref.strip().lower()
            for pred, ref in zip(predictions, references)
        )
        return matches / len(predictions)
    
    def rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            
            scores = {
                'rouge1_precision': [],
                'rouge1_recall': [],
                'rouge1_fmeasure': [],
                'rouge2_precision': [],
                'rouge2_recall': [],
                'rouge2_fmeasure': [],
                'rougeL_precision': [],
                'rougeL_recall': [],
                'rougeL_fmeasure': [],
            }
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                
                for key in ['rouge1', 'rouge2', 'rougeL']:
                    scores[f'{key}_precision'].append(score[key].precision)
                    scores[f'{key}_recall'].append(score[key].recall)
                    scores[f'{key}_fmeasure'].append(score[key].fmeasure)
            
            # Average scores
            return {k: np.mean(v) for k, v in scores.items()}
            
        except ImportError:
            logger.warning("rouge-score not installed. Skipping ROUGE metrics.")
            return {}
    
    def bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            BLEU score (0-1)
        """
        try:
            from sacrebleu import corpus_bleu
            
            # Format references for sacrebleu
            refs = [[ref] for ref in references]
            
            score = corpus_bleu(predictions, refs)
            return score.score / 100.0  # Convert to 0-1 range
            
        except ImportError:
            logger.warning("sacrebleu not installed. Skipping BLEU score.")
            return 0.0
    
    def token_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate token-level accuracy.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Token accuracy (0-1)
        """
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # Use min length to avoid penalizing length differences twice
            min_len = min(len(pred_tokens), len(ref_tokens))
            
            for i in range(min_len):
                total_tokens += 1
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Dictionary of all metrics
        """
        logger.info("Computing evaluation metrics...")
        
        metrics = {}
        
        # Exact match
        metrics['exact_match'] = self.exact_match(predictions, references)
        logger.info(f"Exact Match: {metrics['exact_match']:.2%}")
        
        # ROUGE scores
        rouge = self.rouge_scores(predictions, references)
        metrics.update(rouge)
        if rouge:
            logger.info(f"ROUGE-1 F1: {rouge.get('rouge1_fmeasure', 0):.2%}")
            logger.info(f"ROUGE-L F1: {rouge.get('rougeL_fmeasure', 0):.2%}")
        
        # BLEU score
        metrics['bleu'] = self.bleu_score(predictions, references)
        logger.info(f"BLEU: {metrics['bleu']:.2%}")
        
        # Token accuracy
        metrics['token_accuracy'] = self.token_accuracy(predictions, references)
        logger.info(f"Token Accuracy: {metrics['token_accuracy']:.2%}")
        
        return metrics


def compute_generation_quality(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute generation quality metrics.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Average length
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    metrics['avg_length'] = np.mean(pred_lengths)
    metrics['avg_ref_length'] = np.mean(ref_lengths)
    metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths)
    
    # Diversity (unique tokens ratio)
    all_tokens = []
    for pred in predictions:
        all_tokens.extend(pred.split())
    
    if all_tokens:
        metrics['diversity'] = len(set(all_tokens)) / len(all_tokens)
    else:
        metrics['diversity'] = 0.0
    
    # Repetition rate
    repetition_scores = []
    for pred in predictions:
        tokens = pred.split()
        if len(tokens) > 1:
            unique_bigrams = len(set(zip(tokens[:-1], tokens[1:])))
            total_bigrams = len(tokens) - 1
            repetition_scores.append(1.0 - unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0)
    
    metrics['repetition_rate'] = np.mean(repetition_scores) if repetition_scores else 0.0
    
    # Coherence (simple heuristic: sentence count)
    coherence_scores = []
    for pred in predictions:
        sentences = re.split(r'[.!?]+', pred)
        sentences = [s.strip() for s in sentences if s.strip()]
        # Normalize by length
        coherence_scores.append(min(len(sentences) / max(len(pred.split()) / 20, 1), 1.0))
    
    metrics['coherence'] = np.mean(coherence_scores) if coherence_scores else 0.0
    
    return metrics


def compute_classification_metrics(
    predictions: List[str],
    references: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """
    Compute classification metrics (for categorization tasks).
    
    Args:
        predictions: List of predicted labels
        references: List of true labels
        labels: List of possible labels
        
    Returns:
        Dictionary with precision, recall, F1
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(references, predictions)
    
    # Macro-averaged metrics
    metrics['precision_macro'] = precision_score(
        references, predictions, average='macro', zero_division=0
    )
    metrics['recall_macro'] = recall_score(
        references, predictions, average='macro', zero_division=0
    )
    metrics['f1_macro'] = f1_score(
        references, predictions, average='macro', zero_division=0
    )
    
    # Micro-averaged metrics
    metrics['precision_micro'] = precision_score(
        references, predictions, average='micro', zero_division=0
    )
    metrics['recall_micro'] = recall_score(
        references, predictions, average='micro', zero_division=0
    )
    metrics['f1_micro'] = f1_score(
        references, predictions, average='micro', zero_division=0
    )
    
    # Per-class metrics
    for label in labels:
        label_preds = [1 if p == label else 0 for p in predictions]
        label_refs = [1 if r == label else 0 for r in references]
        
        metrics[f'{label}_precision'] = precision_score(
            label_refs, label_preds, zero_division=0
        )
        metrics[f'{label}_recall'] = recall_score(
            label_refs, label_preds, zero_division=0
        )
        metrics[f'{label}_f1'] = f1_score(
            label_refs, label_preds, zero_division=0
        )
    
    return metrics