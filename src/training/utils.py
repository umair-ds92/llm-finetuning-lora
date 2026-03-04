"""
Training utilities for LoRA fine-tuning.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_training_config(output_dir: str, config: Dict):
    """
    Save training configuration to file.
    
    Args:
        output_dir: Directory to save config
        config: Configuration dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "training_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved training config to {config_path}")


def load_training_config(config_path: str) -> Dict:
    """
    Load training configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded training config from {config_path}")
    
    return config


def get_trainable_parameters(model) -> Dict[str, int]:
    """
    Calculate trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = 0
    all_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
    
    return {
        'trainable_params': trainable_params,
        'all_params': all_params,
        'trainable_percentage': 100 * trainable_params / all_params,
    }


def print_model_info(model):
    """
    Print model information.
    
    Args:
        model: PyTorch model
    """
    param_info = get_trainable_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Total parameters: {param_info['all_params']:,}")
    print(f"Trainable parameters: {param_info['trainable_params']:,}")
    print(f"Trainable %: {param_info['trainable_percentage']:.2f}%")
    print("="*60 + "\n")
    
    logger.info(
        f"Model has {param_info['trainable_params']:,} trainable parameters "
        f"({param_info['trainable_percentage']:.2f}%)"
    )


def estimate_training_time(
    num_examples: int,
    batch_size: int,
    num_epochs: int,
    seconds_per_batch: float = 1.0,
) -> Dict[str, float]:
    """
    Estimate training time.
    
    Args:
        num_examples: Number of training examples
        batch_size: Batch size
        num_epochs: Number of epochs
        seconds_per_batch: Estimated seconds per batch
        
    Returns:
        Dictionary with time estimates
    """
    steps_per_epoch = num_examples // batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * seconds_per_batch
    
    hours = total_seconds / 3600
    minutes = (total_seconds % 3600) / 60
    
    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'total_seconds': total_seconds,
        'hours': hours,
        'minutes': minutes,
    }


def check_gpu_availability() -> Dict[str, any]:
    """
    Check GPU availability and memory.
    
    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'device_name': None,
            'memory_gb': 0,
        }
    
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    
    # Get memory info for first GPU
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        'available': True,
        'count': device_count,
        'device_name': device_name,
        'memory_total_gb': memory_total,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
        'memory_free_gb': memory_total - memory_reserved,
    }


def print_gpu_info():
    """Print GPU information."""
    gpu_info = check_gpu_availability()
    
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    if gpu_info['available']:
        print(f"GPU Available: Yes")
        print(f"GPU Count: {gpu_info['count']}")
        print(f"GPU Name: {gpu_info['device_name']}")
        print(f"Total Memory: {gpu_info['memory_total_gb']:.2f} GB")
        print(f"Free Memory: {gpu_info['memory_free_gb']:.2f} GB")
    else:
        print("GPU Available: No (using CPU)")
    
    print("="*60 + "\n")


def cleanup_checkpoints(
    output_dir: str,
    keep_last_n: int = 3,
):
    """
    Clean up old checkpoints, keeping only the most recent.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
    """
    checkpoint_dirs = []
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            checkpoint_dirs.append(item_path)
    
    # Sort by step number
    checkpoint_dirs.sort(
        key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0
    )
    
    # Remove old checkpoints
    if len(checkpoint_dirs) > keep_last_n:
        import shutil
        for checkpoint_dir in checkpoint_dirs[:-keep_last_n]:
            logger.info(f"Removing old checkpoint: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in output directory.
    
    Args:
        output_dir: Directory to search
        
    Returns:
        Path to latest checkpoint or None
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            try:
                step = int(item.split('-')[-1])
                checkpoints.append((step, item_path))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    # Return checkpoint with highest step number
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint


def verify_checkpoint(checkpoint_path: str) -> bool:
    """
    Verify checkpoint integrity.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        True if checkpoint is valid
    """
    required_files = [
        'adapter_model.bin',  # LoRA weights
        'adapter_config.json',  # LoRA config
    ]
    
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            logger.warning(f"Missing required file: {file}")
            return False
    
    logger.info(f"Checkpoint verified: {checkpoint_path}")
    return True