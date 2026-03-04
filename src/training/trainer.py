"""
Custom trainer class for LoRA fine-tuning with enhanced features.
"""

import os
from typing import Dict, Optional

import torch
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LoRATrainer(Trainer):
    """
    Custom trainer for LoRA fine-tuning with additional features:
    - Enhanced checkpoint saving
    - Custom logging
    - Memory optimization
    - LoRA-specific metrics
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize LoRA trainer."""
        super().__init__(*args, **kwargs)
        
        logger.info("Initialized LoRATrainer")
        
        # Track best metrics
        self.best_metric = None
        self.best_model_checkpoint = None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute training loss.
        
        Override to add custom loss computation if needed.
        """
        return super().compute_loss(model, inputs, return_outputs)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with LoRA adapters only.
        
        This saves only the LoRA adapter weights, not the full model,
        significantly reducing checkpoint size.
        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        # Save LoRA adapters
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir)
            logger.info(f"Saved LoRA adapters to {output_dir}")
        
        # Save optimizer and scheduler
        if self.args.should_save:
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(output_dir, "optimizer.pt")
            )
            
            if self.lr_scheduler is not None:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, "scheduler.pt")
                )
        
        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        # Track best model
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_value = metrics.get(self.args.metric_for_best_model)
            
            if metric_value is not None:
                if (self.best_metric is None or 
                    self._is_better_metric(metric_value, self.best_metric)):
                    self.best_metric = metric_value
                    self.best_model_checkpoint = output_dir
                    logger.info(
                        f"New best model! "
                        f"{self.args.metric_for_best_model}={metric_value:.4f}"
                    )
        
        # Clean up old checkpoints
        self._rotate_checkpoints(use_mtime=False)
        
        return output_dir
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.args.greater_is_better:
            return current > best
        return current < best
    
    def _rotate_checkpoints(self, use_mtime=False):
        """
        Rotate checkpoints to keep only the most recent ones.
        
        Args:
            use_mtime: Whether to use modification time for sorting
        """
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return
        
        # Get all checkpoint directories
        checkpoints = []
        for subdir in os.listdir(self.args.output_dir):
            if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                checkpoint_path = os.path.join(self.args.output_dir, subdir)
                if os.path.isdir(checkpoint_path):
                    checkpoints.append(checkpoint_path)
        
        # Sort checkpoints
        if use_mtime:
            checkpoints.sort(key=lambda x: os.path.getmtime(x))
        else:
            # Sort by step number
            checkpoints.sort(
                key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
            )
        
        # Remove old checkpoints
        if len(checkpoints) > self.args.save_total_limit:
            for checkpoint in checkpoints[:-self.args.save_total_limit]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                self._remove_checkpoint(checkpoint)
    
    def _remove_checkpoint(self, checkpoint_path: str):
        """Remove a checkpoint directory."""
        import shutil
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log training metrics with enhanced formatting.
        
        Args:
            logs: Dictionary of metrics to log
        """
        # Add custom metrics
        if self.state.global_step > 0:
            # Calculate throughput
            if 'loss' in logs:
                logs['learning_rate'] = self._get_learning_rate()
            
            # Add GPU memory usage if available
            if torch.cuda.is_available():
                logs['gpu_memory_allocated_gb'] = (
                    torch.cuda.memory_allocated() / 1024**3
                )
                logs['gpu_memory_reserved_gb'] = (
                    torch.cuda.memory_reserved() / 1024**3
                )
        
        # Call parent log method
        super().log(logs)
    
    def _get_learning_rate(self) -> float:
        """Get current learning rate."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        return self.args.learning_rate
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation with enhanced metrics.
        
        Args:
            eval_dataset: Evaluation dataset
            ignore_keys: Keys to ignore in output
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")
        
        # Run parent evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # Log evaluation summary
        logger.info("Evaluation metrics:")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value}")
        
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call=False):
        """
        Save the LoRA model adapters.
        
        Args:
            output_dir: Directory to save model
            _internal_call: Whether this is an internal call
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving LoRA adapters to {output_dir}")
        
        # Save LoRA adapters
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        logger.info("Model saved successfully")
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        Log, save, and evaluate with custom enhancements.
        """
        # Log current state
        if self.control.should_log:
            logs = {}
            
            # Training loss
            tr_loss_scalar = tr_loss.item() if isinstance(tr_loss, torch.Tensor) else tr_loss
            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4
            )
            logs["learning_rate"] = self._get_learning_rate()
            
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            
            self.log(logs)
        
        # Evaluate
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)
        
        # Save
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=None)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )