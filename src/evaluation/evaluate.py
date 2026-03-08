#!/usr/bin/env python3
"""
Main evaluation script for fine-tuned models.

Evaluates models on:
- Task-specific metrics (accuracy, F1, precision/recall)
- General capability testing
- Comparative analysis (base vs fine-tuned)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data import DatasetLoader
from src.evaluation import (
    TaskMetrics,
    compute_generation_quality,
)
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned LLM models"
    )
    
    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model or HuggingFace model ID",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to LoRA adapter weights (if evaluating fine-tuned model)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model",
        help="Name for this model in reports",
    )
    
    # Data arguments
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test dataset (JSONL)",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    
    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    
    # Comparison
    parser.add_argument(
        "--compare-with-base",
        action="store_true",
        help="Compare fine-tuned model with base model",
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load model (with adapter if specified) and tokenizer."""
    logger.info(f"Loading base model: {args.base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        padding_side="left",  # For generation
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    )
    
    # Load adapter if provided
    if args.adapter_path:
        logger.info(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
        logger.info("Adapter loaded and merged")
    else:
        model = base_model
        logger.info("Using base model without adapter")
    
    model.eval()
    
    return model, tokenizer, base_model if args.adapter_path else None


def load_test_dataset(args, tokenizer):
    """Load test dataset."""
    logger.info(f"Loading test dataset: {args.test_data}")
    
    # Load dataset
    dataset_loader = DatasetLoader(
        tokenizer=tokenizer,
        max_length=512,
    )
    
    test_dataset = dataset_loader.load_from_jsonl(args.test_data)
    
    # Limit samples if specified
    if args.max_samples and args.max_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(args.max_samples))
        logger.info(f"Limited to {args.max_samples} samples for testing")
    
    logger.info(f"Loaded {len(test_dataset)} test examples")
    
    return test_dataset


def evaluate_model(model, tokenizer, test_dataset, args) -> Dict:
    """Evaluate model on test dataset."""
    logger.info("Starting evaluation...")
    
    # Generate predictions
    predictions = []
    references = []
    inputs_list = []
    
    for i, example in enumerate(test_dataset):
        # Create prompt
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        
        # Generate
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        predictions.append(generated.strip())
        references.append(example['output'].strip())
        inputs_list.append({
            'instruction': example['instruction'],
            'input': example['input'],
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(test_dataset)} examples")
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    metrics = TaskMetrics()
    results = metrics.compute_all_metrics(predictions, references)
    
    # Add generation quality metrics
    quality_metrics = compute_generation_quality(predictions, references)
    results.update(quality_metrics)
    
    # Store predictions for analysis
    results['predictions'] = [
        {
            'instruction': inp['instruction'],
            'input': inp['input'],
            'reference': ref,
            'prediction': pred,
        }
        for inp, ref, pred in zip(inputs_list, references, predictions)
    ]
    
    return results


def save_results(results: Dict, output_dir: str, model_name: str):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    results_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to: {results_path}")
    
    # Save summary (without predictions)
    summary = {k: v for k, v in results.items() if k != 'predictions'}
    summary_path = os.path.join(output_dir, f"{model_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to: {summary_path}")
    
    return results_path, summary_path


def print_summary(results: Dict, model_name: str):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS: {model_name}")
    print("="*80)
    
    # Task metrics
    if 'exact_match' in results:
        print(f"\nTask Performance:")
        print(f"  Exact Match:    {results['exact_match']:.2%}")
        print(f"  ROUGE-1 F1:     {results.get('rouge1_fmeasure', 0):.2%}")
        print(f"  ROUGE-2 F1:     {results.get('rouge2_fmeasure', 0):.2%}")
        print(f"  ROUGE-L F1:     {results.get('rougeL_fmeasure', 0):.2%}")
        print(f"  BLEU:           {results.get('bleu', 0):.2%}")
    
    # Generation quality
    if 'avg_length' in results:
        print(f"\nGeneration Quality:")
        print(f"  Avg Length:     {results['avg_length']:.1f} tokens")
        print(f"  Diversity:      {results.get('diversity', 0):.2%}")
        print(f"  Coherence:      {results.get('coherence', 0):.2f}")
    
    print("="*80 + "\n")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer, base_model = load_model_and_tokenizer(args)
    
    # Load test dataset
    test_dataset = load_test_dataset(args, tokenizer)
    
    # Evaluate fine-tuned model
    logger.info(f"Evaluating {args.model_name}...")
    results = evaluate_model(model, tokenizer, test_dataset, args)
    
    # Save and print results
    save_results(results, args.output_dir, args.model_name)
    print_summary(results, args.model_name)
    
    # Compare with base model if requested
    if args.compare_with_base and base_model is not None:
        logger.info("Evaluating base model for comparison...")
        base_results = evaluate_model(base_model, tokenizer, test_dataset, args)
        save_results(base_results, args.output_dir, "base_model")
        print_summary(base_results, "Base Model")
        
        # Compute improvements
        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        
        for metric in ['exact_match', 'rouge1_fmeasure', 'bleu']:
            if metric in results and metric in base_results:
                improvement = results[metric] - base_results[metric]
                print(f"{metric:20s}: {improvement:+.2%}")
        
        print("="*80 + "\n")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()