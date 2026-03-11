#!/usr/bin/env python3
"""
Inference engine for deployed models.

Handles text generation with optimized inference settings.
"""

import argparse
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class StopOnTokens(StoppingCriteria):
    """Stop generation on specific tokens."""
    
    def __init__(self, stop_token_ids: List[int]):
        """
        Initialize stopping criteria.
        
        Args:
            stop_token_ids: List of token IDs to stop on
        """
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if generation should stop."""
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class InferenceEngine:
    """Optimized inference engine for text generation."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype = None,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model
            device: Device to use ('auto', 'cpu', 'cuda')
            torch_dtype: Torch dtype for model
        """
        logger.info(f"Loading model from: {model_path}")
        
        self.device = device
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=False,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        logger.info("Inference engine initialized")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            stop_strings: Strings to stop generation on
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        # Setup stopping criteria
        stopping_criteria = None
        if stop_strings:
            stop_token_ids = []
            for stop_str in stop_strings:
                tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            
            if stop_token_ids:
                stopping_criteria = StoppingCriteriaList([
                    StopOnTokens(stop_token_ids)
                ])
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        # Clean up stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
        
        generation_time = time.time() - start_time
        output_length = outputs.shape[1] - input_length
        tokens_per_second = output_length / generation_time if generation_time > 0 else 0
        
        return {
            'generated_text': generated_text.strip(),
            'input_length': input_length,
            'output_length': output_length,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
        }
    
    def chat(
        self,
        instruction: str,
        input_text: str = "",
        **generation_kwargs
    ) -> Dict[str, any]:
        """
        Generate response in chat format.
        
        Args:
            instruction: Instruction for the model
            input_text: Optional input context
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generation results
        """
        # Format prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Generate
        result = self.generate(
            prompt,
            stop_strings=["###", "\n\n\n"],
            **generation_kwargs
        )
        
        return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test inference engine"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is artificial intelligence?",
        help="Test prompt",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    return parser.parse_args()


def interactive_mode(engine: InferenceEngine):
    """Run inference engine in interactive mode."""
    print("\n" + "="*80)
    print("INTERACTIVE INFERENCE MODE")
    print("="*80)
    print("Enter 'quit' to exit\n")
    
    while True:
        try:
            instruction = input("\nInstruction: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                break
            
            if not instruction:
                continue
            
            input_text = input("Input (optional): ").strip()
            
            print("\nGenerating response...")
            
            result = engine.chat(
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=256,
                temperature=0.7,
            )
            
            print("\n" + "-"*80)
            print("Response:")
            print(result['generated_text'])
            print("-"*80)
            print(f"Generated {result['output_length']} tokens in {result['generation_time']:.2f}s")
            print(f"Speed: {result['tokens_per_second']:.2f} tokens/second")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def main():
    """Main inference function."""
    args = parse_args()
    
    # Initialize engine
    engine = InferenceEngine(args.model_path)
    
    if args.interactive:
        interactive_mode(engine)
    else:
        # Single test
        logger.info(f"Testing with prompt: {args.prompt}")
        
        result = engine.generate(
            args.prompt,
            max_new_tokens=100,
            temperature=0.7,
        )
        
        print("\n" + "="*80)
        print("INFERENCE TEST")
        print("="*80)
        print(f"Prompt: {args.prompt}")
        print(f"\nGenerated:")
        print(result['generated_text'])
        print("\n" + "-"*80)
        print(f"Input length:    {result['input_length']} tokens")
        print(f"Output length:   {result['output_length']} tokens")
        print(f"Generation time: {result['generation_time']:.2f} seconds")
        print(f"Speed:           {result['tokens_per_second']:.2f} tokens/second")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
