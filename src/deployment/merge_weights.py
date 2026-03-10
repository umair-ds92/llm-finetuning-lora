#!/usr/bin/env python3
"""
Merge LoRA adapter weights with base model for deployment.

This script merges LoRA adapters back into the base model to create
a single, deployment-ready model file.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights with base model"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model or HuggingFace model ID",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for merging",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision after merging",
    )
    
    return parser.parse_args()


def get_torch_dtype(precision: str):
    """Convert precision string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(precision, torch.float16)


def merge_lora_weights(args):
    """
    Merge LoRA adapter weights with base model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Merged model and tokenizer
    """
    logger.info(f"Loading base model: {args.base_model}")
    
    # Determine dtype
    dtype = get_torch_dtype(args.precision)
    logger.info(f"Using precision: {args.precision} ({dtype})")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=args.device,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    
    logger.info(f"Loading LoRA adapter from: {args.adapter_path}")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    logger.info("Merging LoRA weights with base model...")
    
    # Merge weights
    merged_model = model.merge_and_unload()
    
    logger.info("LoRA weights merged successfully")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Calculate model size
    param_count = sum(p.numel() for p in merged_model.parameters())
    logger.info(f"Merged model parameters: {param_count:,}")
    
    return merged_model, tokenizer


def save_merged_model(model, tokenizer, output_dir: str):
    """
    Save merged model and tokenizer.
    
    Args:
        model: Merged model
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving merged model to: {output_dir}")
    
    # Save model
    model.save_pretrained(
        output_dir,
        safe_serialization=True,  # Use safetensors format
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model and tokenizer saved successfully")
    
    # Print directory contents
    model_files = list(Path(output_dir).glob("*"))
    total_size = sum(f.stat().st_size for f in model_files if f.is_file())
    
    logger.info(f"Total model size: {total_size / 1024**3:.2f} GB")
    logger.info(f"Files saved: {len(model_files)}")


def verify_merged_model(output_dir: str):
    """
    Verify merged model can be loaded.
    
    Args:
        output_dir: Directory containing merged model
        
    Returns:
        True if verification successful
    """
    logger.info("Verifying merged model...")
    
    try:
        # Try loading the model
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Quick inference test
        test_input = "Hello, this is a test."
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info("✓ Merged model verification successful")
        logger.info(f"Test generation: {result[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Merged model verification failed: {e}")
        return False


def main():
    """Main merging function."""
    args = parse_args()
    
    # Merge weights
    merged_model, tokenizer = merge_lora_weights(args)
    
    # Save merged model
    save_merged_model(merged_model, tokenizer, args.output_dir)
    
    # Verify
    if verify_merged_model(args.output_dir):
        logger.info("✓ Weight merging complete!")
        
        print("\n" + "="*80)
        print("WEIGHT MERGING SUMMARY")
        print("="*80)
        print(f"Base model:      {args.base_model}")
        print(f"Adapter:         {args.adapter_path}")
        print(f"Output:          {args.output_dir}")
        print(f"Precision:       {args.precision}")
        print("Status:          ✓ SUCCESS")
        print("="*80 + "\n")
        
        print("Next steps:")
        print("1. Quantize the model (optional):")
        print(f"   python src/deployment/quantize.py --model-path {args.output_dir}")
        print("\n2. Test the merged model:")
        print(f"   python src/deployment/inference.py --model-path {args.output_dir}")
        print("\n3. Deploy with FastAPI:")
        print(f"   python src/deployment/serve.py --model-path {args.output_dir}")
    else:
        logger.error("✗ Weight merging failed verification")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
