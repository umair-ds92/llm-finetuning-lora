#!/usr/bin/env python3
"""
Main LoRA fine-tuning script using PEFT and DeepSpeed.

This script implements the complete training pipeline with:
- LoRA parameter-efficient fine-tuning
- DeepSpeed ZeRO-2 optimization
- Checkpoint management
- WandB/TensorBoard logging
- Learning rate scheduling
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.data import DatasetLoader, validate_dataset_format
from src.training import LoRATrainer, get_training_callbacks
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs with LoRA using PEFT"
    )
    
    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model or HuggingFace model ID",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama-2", "mistral"],
        default="llama-2",
        help="Model architecture type",
    )
    
    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        help="Path to validation data (JSONL)",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability",
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/lora-finetuned",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (overrides CLI args)",
    )
    
    # Optimization
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Use DeepSpeed for training",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="configs/deepspeed_config.json",
        help="DeepSpeed configuration file",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    
    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="llm-finetuning-lora",
        help="WandB project name",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    
    # Checkpointing
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint to resume from",
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_config_with_args(args, config: dict) -> argparse.Namespace:
    """Merge YAML config with command line arguments."""
    # Config file takes precedence
    if config:
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return args


def setup_model_and_tokenizer(args):
    """Load base model and tokenizer, apply LoRA."""
    logger.info(f"Loading base model: {args.base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        padding_side="right",
    )
    
    # Add pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=False,
    )
    
    # Prepare for training
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Can be configured
    )
    
    logger.info(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    
    return model, tokenizer


def setup_datasets(args, tokenizer):
    """Load and prepare training datasets."""
    logger.info("Loading datasets...")
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(
        tokenizer=tokenizer,
        max_length=2048,
    )
    
    # Load training data
    train_dataset = dataset_loader.load_from_jsonl(args.train_data)
    validate_dataset_format(train_dataset)
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    
    # Load validation data if provided
    eval_dataset = None
    if args.val_data:
        eval_dataset = dataset_loader.load_from_jsonl(args.val_data)
        validate_dataset_format(eval_dataset)
        logger.info(f"Loaded {len(eval_dataset)} validation examples")
    
    # Tokenize datasets
    train_dataset = dataset_loader.prepare_dataset(train_dataset)
    
    if eval_dataset:
        eval_dataset = dataset_loader.prepare_dataset(eval_dataset)
    
    return train_dataset, eval_dataset


def setup_training_args(args):
    """Configure training arguments."""
    # Determine report_to based on flags
    report_to = []
    if args.use_wandb:
        report_to.append("wandb")
    report_to.append("tensorboard")
    
    training_args = TrainingArguments(
        # Output
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        
        # Training
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimization
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        
        # Precision
        fp16=False,
        bf16=args.bf16,
        tf32=True,
        
        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=report_to,
        
        # Evaluation
        evaluation_strategy="steps" if args.val_data else "no",
        eval_steps=500 if args.val_data else None,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        
        # DeepSpeed
        deepspeed=args.deepspeed_config if args.use_deepspeed else None,
        
        # Other
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=False,
        ddp_find_unused_parameters=False,
        
        # Resume
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    return training_args


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config if provided
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        args = merge_config_with_args(args, config.get('training', {}))
    
    # Set seed
    set_seed(args.seed)
    
    # Setup WandB if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"lora-r{args.lora_r}-lr{args.learning_rate}",
            config=vars(args),
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log arguments
    logger.info("Training arguments:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(args, tokenizer)
    
    # Setup training arguments
    training_args = setup_training_args(args)
    
    # Setup callbacks
    callbacks = get_training_callbacks(args)
    
    # Initialize trainer
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save training state
    trainer.save_state()
    
    logger.info("Training complete!")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total steps: {metrics.get('train_steps', 'N/A')}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Training runtime: {metrics.get('train_runtime', 'N/A'):.2f}s")
    print(f"Samples per second: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()