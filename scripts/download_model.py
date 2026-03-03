#!/usr/bin/env python3
"""
Script to download base models from HuggingFace Hub.

This script:
1. Downloads specified model from HuggingFace
2. Verifies model can be loaded
3. Saves model locally for training
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import setup_logger

logger = setup_logger(__name__, log_file="logs/model_download.log")


SUPPORTED_MODELS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
}


def download_model(
    model_name: str,
    output_dir: Path,
    use_auth_token: str = None,
) -> bool:
    """
    Download model from HuggingFace Hub.
    
    Args:
        model_name: Model identifier
        output_dir: Directory to save model
        use_auth_token: HuggingFace authentication token
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            token=use_auth_token,
            ignore_patterns=["*.safetensors"],  # Download PyTorch weights
        )
        
        logger.info(f"✓ Model downloaded successfully to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download model: {str(e)}")
        return False


def verify_model(
    model_path: Path,
    use_auth_token: str = None,
) -> bool:
    """
    Verify that model can be loaded.
    
    Args:
        model_path: Path to model directory
        use_auth_token: HuggingFace authentication token
        
    Returns:
        True if model loads successfully
    """
    try:
        logger.info(f"Verifying model at {model_path}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_auth_token=use_auth_token,
        )
        logger.info(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
        
        # Load model (on CPU for verification)
        logger.info("Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="cpu",
            torch_dtype="auto",
            use_auth_token=use_auth_token,
            low_cpu_mem_usage=True,
        )
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {num_params:,}")
        logger.info(f"  Config: {model.config}")
        
        # Test inference
        logger.info("Testing inference...")
        test_input = "Hello, world!"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✓ Inference test passed")
        logger.info(f"  Input: {test_input}")
        logger.info(f"  Output: {generated_text}")
        
        # Clean up
        del model
        del tokenizer
        import torch
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model verification failed: {str(e)}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download and verify base models for fine-tuning"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()) + list(SUPPORTED_MODELS.values()),
        help="Model to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="HuggingFace authentication token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only verify existing model",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip verification after download",
    )
    
    args = parser.parse_args()
    
    # Get authentication token
    auth_token = args.auth_token or os.getenv("HF_TOKEN")
    if auth_token:
        logger.info("Using HuggingFace authentication token")
    
    # Resolve model name
    if args.model_name in SUPPORTED_MODELS:
        model_id = SUPPORTED_MODELS[args.model_name]
        short_name = args.model_name
    else:
        model_id = args.model_name
        short_name = model_id.split("/")[-1]
    
    # Set output directory
    output_dir = Path(args.output_dir) / short_name
    
    # Download model
    if not args.skip_download:
        success = download_model(model_id, output_dir, auth_token)
        if not success:
            logger.error("Model download failed")
            return 1
    
    # Verify model
    if not args.skip_verification:
        success = verify_model(output_dir, auth_token)
        if not success:
            logger.error("Model verification failed")
            return 1
    
    logger.info("✓ All checks passed!")
    logger.info(f"Model ready for training at: {output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    import torch  # Import here for verification
    sys.exit(main())