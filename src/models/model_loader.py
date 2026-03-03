"""
Model loading and configuration utilities.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Load and configure models for fine-tuning."""
    
    def __init__(
        self,
        model_name_or_path: str,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize model loader.
        
        Args:
            model_name_or_path: Model name or path
            use_auth_token: HuggingFace authentication token
            cache_dir: Directory to cache models
        """
        self.model_name_or_path = model_name_or_path
        self.use_auth_token = use_auth_token
        self.cache_dir = cache_dir
        
        logger.info(f"Initialized ModelLoader for {model_name_or_path}")
    
    def load_tokenizer(
        self,
        padding_side: str = "right",
        add_eos_token: bool = True,
    ) -> PreTrainedTokenizer:
        """
        Load tokenizer for the model.
        
        Args:
            padding_side: Padding side ("left" or "right")
            add_eos_token: Whether to add EOS token
            
        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer from {self.model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_auth_token=self.use_auth_token,
            cache_dir=self.cache_dir,
            padding_side=padding_side,
        )
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        if add_eos_token:
            tokenizer.add_eos_token = True
        
        logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer)}")
        
        return tokenizer
    
    def load_base_model(
        self,
        torch_dtype: Union[str, torch.dtype] = "auto",
        device_map: str = "auto",
        low_cpu_mem_usage: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> PreTrainedModel:
        """
        Load base model for fine-tuning.
        
        Args:
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            low_cpu_mem_usage: Whether to use low CPU memory mode
            load_in_8bit: Whether to load in 8-bit quantization
            load_in_4bit: Whether to load in 4-bit quantization
            
        Returns:
            Loaded base model
        """
        logger.info(f"Loading base model from {self.model_name_or_path}")
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else False,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
            )
            logger.info(
                f"Using {'8-bit' if load_in_8bit else '4-bit'} quantization"
            )
        
        # Convert torch_dtype string to dtype
        if isinstance(torch_dtype, str):
            if torch_dtype == "auto":
                torch_dtype = torch.float16
            elif torch_dtype == "bfloat16" or torch_dtype == "bf16":
                torch_dtype = torch.bfloat16
            elif torch_dtype == "float16" or torch_dtype == "fp16":
                torch_dtype = torch.float16
            elif torch_dtype == "float32" or torch_dtype == "fp32":
                torch_dtype = torch.float32
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            use_auth_token=self.use_auth_token,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=False,
        )
        
        # Prepare model for k-bit training if using quantization
        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model)
            logger.info("Prepared model for k-bit training")
        
        # Get model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded model with {num_params:,} parameters")
        
        return model
    
    def configure_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ) -> LoraConfig:
        """
        Configure LoRA parameters.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: Dropout probability
            target_modules: List of module names to apply LoRA
            bias: Bias training strategy ("none", "all", "lora_only")
            task_type: Task type for the model
            
        Returns:
            LoRA configuration
        """
        # Default target modules for Llama/Mistral
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=task_type,
        )
        
        logger.info(
            f"Configured LoRA: r={r}, alpha={lora_alpha}, "
            f"dropout={lora_dropout}, targets={target_modules}"
        )
        
        return lora_config
    
    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoraConfig,
    ) -> PeftModel:
        """
        Apply LoRA to base model.
        
        Args:
            model: Base model
            lora_config: LoRA configuration
            
        Returns:
            Model with LoRA adapters
        """
        logger.info("Applying LoRA to model")
        
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params
        
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_percent:.2f}%)"
        )
        
        return model
    
    def load_model_for_training(
        self,
        lora_config: Optional[LoraConfig] = None,
        **model_kwargs,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer ready for training.
        
        Args:
            lora_config: LoRA configuration (if None, creates default)
            **model_kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = self.load_tokenizer()
        
        # Load base model
        model = self.load_base_model(**model_kwargs)
        
        # Apply LoRA if config provided
        if lora_config is None:
            lora_config = self.configure_lora()
        
        model = self.apply_lora(model, lora_config)
        
        # Enable gradient checkpointing for memory efficiency
        model.config.use_cache = False
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        
        logger.info("Model and tokenizer ready for training")
        
        return model, tokenizer


def verify_model_loading(
    model_name_or_path: str,
    use_auth_token: Optional[str] = None,
) -> bool:
    """
    Verify that a model can be loaded successfully.
    
    Args:
        model_name_or_path: Model name or path
        use_auth_token: HuggingFace authentication token
        
    Returns:
        True if model loads successfully
    """
    try:
        logger.info(f"Verifying model loading: {model_name_or_path}")
        
        # Try loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_auth_token=use_auth_token,
        )
        logger.info(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
        
        # Try loading model config
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cpu",  # Load on CPU for verification
            torch_dtype=torch.float16,
            use_auth_token=use_auth_token,
            low_cpu_mem_usage=True,
        )
        logger.info(f"✓ Model loaded successfully")
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {str(e)}")
        return False