"""
Training callbacks for monitoring and control.
"""

import time
from typing import Dict, List

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage."""
    
    def __init__(self):
        """Initialize memory callback."""
        self.start_time = None
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Track training start time."""
        self.start_time = time.time()
        logger.info("Training started")
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log memory usage periodically."""
        if state.global_step % 100 == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(
                        f"Step {state.global_step}: "
                        f"GPU memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved"
                    )
            except Exception as e:
                logger.warning(f"Could not log GPU memory: {e}")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log training completion."""
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Training completed in {duration:.2f} seconds")


class ProgressCallback(TrainerCallback):
    """Callback to log training progress."""
    
    def __init__(self, total_steps: int = None):
        """
        Initialize progress callback.
        
        Args:
            total_steps: Total number of training steps
        """
        self.total_steps = total_steps
        self.last_log_time = None
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log progress at intervals."""
        if state.global_step % args.logging_steps == 0:
            current_time = time.time()
            
            # Calculate progress
            if self.total_steps:
                progress = (state.global_step / self.total_steps) * 100
                logger.info(
                    f"Progress: {state.global_step}/{self.total_steps} "
                    f"({progress:.1f}%)"
                )
            
            # Calculate speed
            if self.last_log_time:
                time_diff = current_time - self.last_log_time
                steps_diff = args.logging_steps
                speed = steps_diff / time_diff
                logger.info(f"Speed: {speed:.2f} steps/second")
            
            self.last_log_time = current_time


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping based on validation metrics."""
    
    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
    ):
        """
        Initialize early stopping callback.
        
        Args:
            early_stopping_patience: Number of evaluations to wait
            early_stopping_threshold: Minimum improvement threshold
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        self.best_metric = None
        self.patience_counter = 0
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """Check early stopping condition."""
        metric_value = metrics.get(args.metric_for_best_model)
        
        if metric_value is None:
            return
        
        # Check if metric improved
        if self.best_metric is None:
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            # Check improvement
            if args.greater_is_better:
                improved = metric_value > (
                    self.best_metric + self.early_stopping_threshold
                )
            else:
                improved = metric_value < (
                    self.best_metric - self.early_stopping_threshold
                )
            
            if improved:
                self.best_metric = metric_value
                self.patience_counter = 0
                logger.info(f"Metric improved to {metric_value:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement. Patience: "
                    f"{self.patience_counter}/{self.early_stopping_patience}"
                )
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    control.should_training_stop = True


class CheckpointCallback(TrainerCallback):
    """Enhanced checkpoint management."""
    
    def __init__(self, save_best_only: bool = False):
        """
        Initialize checkpoint callback.
        
        Args:
            save_best_only: Only save checkpoints that improve metrics
        """
        self.save_best_only = save_best_only
        self.best_metric = None
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Handle checkpoint saving."""
        logger.info(f"Saving checkpoint at step {state.global_step}")
        
        if self.save_best_only and state.best_metric is not None:
            if (self.best_metric is None or 
                state.best_metric > self.best_metric):
                self.best_metric = state.best_metric
                logger.info(f"Saving best model (metric: {self.best_metric:.4f})")
            else:
                logger.info("Skipping save (not best model)")
                control.should_save = False


class LossLoggingCallback(TrainerCallback):
    """Detailed loss logging."""
    
    def __init__(self):
        """Initialize loss logging callback."""
        self.losses = []
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs,
    ):
        """Log detailed loss information."""
        if 'loss' in logs:
            loss = logs['loss']
            self.losses.append(loss)
            
            # Calculate moving average
            if len(self.losses) >= 10:
                moving_avg = sum(self.losses[-10:]) / 10
                logger.info(f"Loss (10-step avg): {moving_avg:.4f}")


def get_training_callbacks(args) -> List[TrainerCallback]:
    """
    Get list of training callbacks based on configuration.
    
    Args:
        args: Training arguments
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Always add memory and progress callbacks
    callbacks.append(MemoryCallback())
    callbacks.append(ProgressCallback())
    callbacks.append(LossLoggingCallback())
    
    # Add early stopping if validation is enabled
    if hasattr(args, 'val_data') and args.val_data:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        )
    
    # Add checkpoint callback
    callbacks.append(CheckpointCallback(save_best_only=False))
    
    logger.info(f"Initialized {len(callbacks)} training callbacks")
    
    return callbacks