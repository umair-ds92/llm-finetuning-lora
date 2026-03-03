"""
LLM Fine-Tuning with PEFT (LoRA)

Enterprise-grade fine-tuning pipeline for adapting open-source LLMs
to domain-specific tasks using Parameter-Efficient Fine-Tuning.
"""

__version__ = "0.1.0"
__author__ = "Muhammad Umair"
__license__ = "Apache 2.0"

from src.utils.logging import setup_logger

# Initialize default logger
logger = setup_logger(__name__)