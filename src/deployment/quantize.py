#!/usr/bin/env python3
"""
Quantize model for efficient deployment.

Supports 4-bit and 8-bit quantization using bitsandbytes.
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantize model for deployment"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model to quantize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=["4bit", "8bit"],
        help="Quantization level (4bit or 8bit)",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Computation dtype for quantized model",
    )
    
    return parser.parse_args()


def get_quantization_config(args):
    """
    Create quantization configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        BitsAndBytesConfig
    """
    compute_dtype = torch.float16 if args.compute_dtype == "float16" else torch.bfloat16
    
    if args.quantization == "4bit":
        logger.info("Configuring 4-bit quantization")
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:  # 8bit
        logger.info("Configuring 8-bit quantization")
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    
    return config


def quantize_model(args):
    """
    Quantize model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Quantized model and tokenizer
    """
    logger.info(f"Loading model from: {args.model_path}")
    
    # Get quantization config
    quant_config = get_quantization_config(args)
    
    # Load model with quantization
    logger.info(f"Applying {args.quantization} quantization...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    logger.info("Model quantized successfully")
    
    # Calculate memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU memory usage: {memory_mb:.2f} MB")
    
    return model, tokenizer


def save_quantized_model(model, tokenizer, output_dir: str):
    """
    Save quantized model.
    
    Args:
        model: Quantized model
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving quantized model to: {output_dir}")
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Quantized model saved successfully")


def test_quantized_model(model, tokenizer):
    """
    Test quantized model with sample inference.
    
    Args:
        model: Quantized model
        tokenizer: Tokenizer
    """
    logger.info("Testing quantized model...")
    
    test_prompt = "### Instruction:\nTest the model\n\n### Response:\n"
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info("✓ Quantized model test successful")
    logger.info(f"Sample output: {result[:100]}...")


def estimate_size_reduction(original_path: str, quantization: str):
    """
    Estimate size reduction from quantization.
    
    Args:
        original_path: Path to original model
        quantization: Quantization level
        
    Returns:
        Estimated reduction info
    """
    try:
        from pathlib import Path
        
        # Calculate original size
        model_files = list(Path(original_path).glob("*.bin")) + \
                     list(Path(original_path).glob("*.safetensors"))
        
        if model_files:
            original_size_gb = sum(f.stat().st_size for f in model_files) / 1024**3
            
            # Estimate quantized size
            if quantization == "4bit":
                reduction_factor = 4
            else:  # 8bit
                reduction_factor = 2
            
            quantized_size_gb = original_size_gb / reduction_factor
            
            return {
                'original_size': original_size_gb,
                'quantized_size': quantized_size_gb,
                'reduction_factor': reduction_factor,
            }
    except Exception as e:
        logger.warning(f"Could not estimate size: {e}")
    
    return None


def main():
    """Main quantization function."""
    args = parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("No GPU detected. Quantization works best on GPU.")
        logger.warning("Proceeding with CPU (may be slow)...")
    
    # Estimate size reduction
    size_info = estimate_size_reduction(args.model_path, args.quantization)
    if size_info:
        logger.info(f"Original model size: {size_info['original_size']:.2f} GB")
        logger.info(f"Expected quantized size: {size_info['quantized_size']:.2f} GB")
        logger.info(f"Reduction factor: {size_info['reduction_factor']}x")
    
    # Quantize model
    model, tokenizer = quantize_model(args)
    
    # Test quantized model
    test_quantized_model(model, tokenizer)
    
    # Save quantized model
    save_quantized_model(model, tokenizer, args.output_dir)
    
    print("\n" + "="*80)
    print("QUANTIZATION SUMMARY")
    print("="*80)
    print(f"Original model:  {args.model_path}")
    print(f"Quantization:    {args.quantization}")
    print(f"Output:          {args.output_dir}")
    if size_info:
        print(f"Size reduction:  {size_info['original_size']:.2f} GB → {size_info['quantized_size']:.2f} GB")
    print("Status:          ✓ SUCCESS")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("1. Test the quantized model:")
    print(f"   python src/deployment/inference.py --model-path {args.output_dir}")
    print("\n2. Deploy with FastAPI:")
    print(f"   python src/deployment/serve.py --model-path {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
